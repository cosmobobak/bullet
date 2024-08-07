use crate::{
    inputs::InputType,
    loader::GpuDataLoader,
    lr::LrScheduler,
    optimiser::Optimiser,
    outputs::OutputBuckets,
    tensor::{device_name, device_synchronise},
    util, LocalSettings, Trainer, TrainingSchedule,
};

use std::{
    fs::File,
    io::{stdout, BufRead, BufReader, Write},
    sync::{
        atomic::{AtomicBool, Ordering::SeqCst},
        mpsc::{sync_channel, Receiver},
    },
    time::Instant,
};

use super::schedule::wdl::WdlScheduler;

#[allow(clippy::too_many_arguments)]
pub fn run<T: InputType, U: OutputBuckets<T::RequiredDataType>, O: Optimiser, F, LR: LrScheduler, WDL: WdlScheduler>(
    trainer: &mut Trainer<T, U, O>,
    schedule: &TrainingSchedule<O::AdditionalOptimiserParams, LR, WDL>,
    settings: &LocalSettings,
    mut callback: F,
) where
    F: FnMut(usize, &Trainer<T, U, O>, &TrainingSchedule<O::AdditionalOptimiserParams, LR, WDL>, &LocalSettings),
{
    let threads = settings.threads;
    let data_file_paths: Vec<_> = settings.data_file_paths.iter().map(|s| s.to_string()).collect();
    let out_dir = settings.output_directory.to_string();
    let out_dir = out_dir.as_str();

    std::fs::create_dir(out_dir).unwrap_or(());

    device_synchronise();

    trainer.set_batch_size(schedule.batch_size);
    trainer.set_ft_reg(schedule.ft_regularisation);

    let data_size = std::mem::size_of::<T::RequiredDataType>() as u64;
    let esc = esc();
    let mut file_size = 0;
    for file in data_file_paths.iter() {
        let this_size = std::fs::metadata(file).unwrap_or_else(|_| panic!("Invalid File Metadata: {file}")).len();

        if this_size % data_size != 0 {
            panic!("File [{file}] does not have a multiple of {data_size} size!");
        }

        file_size += this_size;
    }

    let num = (file_size / data_size) as usize;
    let batch_size = trainer.batch_size();

    if device_name() == "CPU" {
        println!("{}", ansi("========== WARNING ==========", 31));
        println!("This backend is not currently");
        println!("   intended to be used for   ");
        println!("  serious training, you may  ");
        println!("  have meant to enable the   ");
        println!("      `cuda` feature.        ");
        println!("{}", ansi("=============================", 31));
    }

    print!("{esc}");
    println!("{}", ansi("Beginning Training", "34;1"));
    println!("Net Name               : {}", ansi(schedule.net_id.clone(), "32;1"));
    println!("Arch                   : {}", ansi(format!("{trainer}"), 31));
    schedule.display();
    println!("Device                 : {}", ansi(device_name(), 31));
    settings.display();
    println!("Positions              : {}", ansi(num, 31));

    let pos_per_sb = schedule.batch_size * schedule.batches_per_superbatch;
    let total_pos = pos_per_sb * (schedule.end_superbatch - schedule.start_superbatch + 1);
    let iters = total_pos as f64 / num as f64;
    println!("Total Epochs           : {}", ansi(format!("{iters:.2}"), 31));

    let timer = Instant::now();

    trainer.set_threads(threads);
    device_synchronise();

    let x = trainer.input_getter();
    let y = trainer.bucket_getter();
    let (sender, reciever) = sync_channel::<GpuDataLoader<T, U>>(512);

    let dataloader =
        create_dataloader::<T, U, O, LR, WDL>(schedule.clone(), data_file_paths, batch_size, x, y, threads, sender);

    let validation_freq = settings.test_set.map_or(1, |test| test.freq);
    let (test_dataloader, test_reciever) = settings
        .test_set
        .map(|test| {
            let file_paths = vec![test.path.to_string()];
            let (sender, reciever) = sync_channel::<GpuDataLoader<T, U>>(512);
            let dataloader = create_dataloader::<T, U, O, LR, WDL>(
                schedule.for_validation(validation_freq),
                file_paths,
                batch_size,
                x,
                y,
                threads,
                sender,
            );
            (dataloader, reciever)
        })
        .unzip();

    let mut prev_lr = schedule.lr(0, 1);
    let mut superbatch = schedule.start_superbatch;
    let mut curr_batch = 0;
    let mut superbatch_timer = Instant::now();
    let optimiser_settings = schedule.optimiser_settings.clone();
    trainer.set_error_zero();

    while let Ok(gpu_loader) = reciever.recv() {
        let lrate = schedule.lr(curr_batch, superbatch);
        if lrate != prev_lr {
            println!("LR Dropped to {}", ansi(lrate, num_cs()));
        }
        prev_lr = lrate;

        trainer.clear_data();
        device_synchronise();

        trainer.load_data(&gpu_loader);
        device_synchronise();

        let valid = trainer.train_on_batch(lrate, schedule.power(), superbatch, curr_batch, &optimiser_settings);
        device_synchronise();

        if !valid {
            trainer.save(out_dir, format!("error-nan-batch-{curr_batch}"));
            panic!("Batch {curr_batch} NaN!");
        }

        // Track test loss every freq batches.
        if curr_batch % validation_freq == 0 {
            if let Some(Ok(test_batch)) = test_reciever.as_ref().map(Receiver::recv) {
                trainer.clear_data();
                device_synchronise();

                trainer.load_data(&test_batch);
                device_synchronise();

                let valid = trainer.evaluate_on_batch(schedule.power(), superbatch, curr_batch);
                device_synchronise();

                if !valid {
                    trainer.save(out_dir, format!("test-error-nan-batch-{curr_batch}"));
                    panic!("Test-set NaN at Batch {curr_batch}!");
                }
            }
        }

        if curr_batch % 128 == 0 {
            report_superbatch_progress(
                superbatch,
                batch_size,
                schedule.batches_per_superbatch,
                curr_batch,
                &superbatch_timer,
            );
        }

        curr_batch += 1;

        if curr_batch % schedule.batches_per_superbatch == 0 {
            let error = trainer.error() / schedule.batches_per_superbatch as f32;

            report_superbatch_finished::<O, LR, WDL>(
                schedule,
                superbatch,
                error,
                &superbatch_timer,
                &timer,
                pos_per_sb,
            );

            callback(superbatch, trainer, schedule, settings);

            superbatch += 1;
            curr_batch = 0;
            superbatch_timer = Instant::now();
            trainer.set_error_zero();
        }
    }

    dataloader.join().unwrap();
    if let Some(h) = test_dataloader {
        if !h.is_finished() {
            println!("Warning: Training set exhausted but test set is not!");
        }
        h.join().unwrap();
    };
}

fn create_dataloader<
    T: InputType,
    U: OutputBuckets<T::RequiredDataType>,
    O: Optimiser,
    LR: LrScheduler,
    WDL: WdlScheduler,
>(
    schedule: TrainingSchedule<<O as Optimiser>::AdditionalOptimiserParams, LR, WDL>,
    data_file_paths: Vec<String>,
    batch_size: usize,
    x: T,
    y: U,
    threads: usize,
    sender: std::sync::mpsc::SyncSender<GpuDataLoader<T, U>>,
) -> std::thread::JoinHandle<()> {
    let buffer_size_mb = 256;
    let buffer_size = buffer_size_mb * 1024 * 1024;
    let data_size: usize = std::mem::size_of::<T::RequiredDataType>();
    let batches_per_load = buffer_size / data_size / batch_size;
    let cap = data_size * batch_size * batches_per_load;
    let rscale = 1.0 / schedule.eval_scale;

    std::thread::spawn(move || {
        let mut curr_superbatch = schedule.start_superbatch;
        let mut curr_batch = 0;

        'dataloading: loop {
            let mut loader_files = vec![];
            for file in data_file_paths.iter() {
                loader_files.push(File::open(file).unwrap_or_else(|_| panic!("Invalid File Path: {file}")));
            }

            for loader_file in loader_files.iter() {
                let mut file = BufReader::with_capacity(cap, loader_file);
                while let Ok(buf) = file.fill_buf() {
                    if buf.is_empty() {
                        break;
                    }

                    let data: &[T::RequiredDataType] = util::to_slice_with_lifetime(buf);

                    for batch in data.chunks(batch_size) {
                        // get wdl blend from the scheduler.
                        let blend = schedule.wdl_scheduler.blend(curr_batch, curr_superbatch, schedule.end_superbatch);
                        let mut gpu_loader = GpuDataLoader::<T, U>::new(x, y);
                        gpu_loader.load(batch, threads, blend, rscale);
                        sender.send(gpu_loader).unwrap();
                        curr_batch += 1;
                        if curr_batch % schedule.batches_per_superbatch == 0 {
                            if curr_superbatch == schedule.end_superbatch {
                                break 'dataloading;
                            }

                            curr_batch = 0;
                            curr_superbatch += 1;
                        }
                    }

                    let consumed = buf.len();
                    file.consume(consumed);
                }
            }
        }
    })
}

static CBCS: AtomicBool = AtomicBool::new(false);

pub fn ansi<T, U>(x: T, y: U) -> String
where
    T: std::fmt::Display,
    U: std::fmt::Display,
{
    format!("\x1b[{}m{}\x1b[0m{}", y, x, esc())
}

pub fn set_cbcs(val: bool) {
    CBCS.store(val, SeqCst)
}

fn num_cs() -> i32 {
    if CBCS.load(SeqCst) {
        35
    } else {
        36
    }
}

fn esc() -> &'static str {
    if CBCS.load(SeqCst) {
        "\x1b[38;5;225m"
    } else {
        ""
    }
}

fn report_superbatch_progress(
    superbatch: usize,
    batch_size: usize,
    batches: usize,
    finished_batches: usize,
    superbatch_timer: &Instant,
) {
    let num_cs = num_cs();
    let superbatch_time = superbatch_timer.elapsed().as_secs_f32();
    let pct = finished_batches as f32 / batches as f32;
    let positions = finished_batches * batch_size;
    let pos_per_sec = positions as f32 / superbatch_time;

    let seconds = superbatch_time / pct - superbatch_time;

    print!(
        "superbatch {} [{}% ({}/{} batches, {} pos/sec)]\n\
        Estimated time to end of superbatch: {}s     \x1b[F",
        ansi(superbatch, num_cs),
        ansi(format!("{:.1}", pct * 100.0), 35),
        ansi(finished_batches, num_cs),
        ansi(batches, num_cs),
        ansi(format!("{pos_per_sec:.0}"), num_cs),
        ansi(format!("{seconds:.1}"), num_cs),
    );
    let _ = stdout().flush();
}

fn report_superbatch_finished<O: Optimiser, LR: LrScheduler, WDL: WdlScheduler>(
    schedule: &TrainingSchedule<O::AdditionalOptimiserParams, LR, WDL>,
    superbatch: usize,
    error: f32,
    superbatch_timer: &Instant,
    timer: &Instant,
    positions: usize,
) {
    let num_cs = num_cs();
    let superbatch_time = superbatch_timer.elapsed().as_secs_f32();
    let total_time = timer.elapsed().as_secs_f32();
    let pos_per_sec = positions as f32 / superbatch_time;

    println!(
        "superbatch {} | time {}s | running loss {} | {} pos/sec | total time {}s",
        ansi(superbatch, num_cs),
        ansi(format!("{superbatch_time:.1}"), num_cs),
        ansi(format!("{error:.6}"), num_cs),
        ansi(format!("{:.0}", pos_per_sec), num_cs),
        ansi(format!("{total_time:.1}"), num_cs),
    );

    let finished_superbatches = superbatch - schedule.start_superbatch + 1;
    let total_superbatches = schedule.end_superbatch - schedule.start_superbatch + 1;
    let pct = finished_superbatches as f32 / total_superbatches as f32;
    let mut seconds = (total_time / pct - total_time) as u32;
    let mut minutes = seconds / 60;
    let hours = minutes / 60;
    seconds -= minutes * 60;
    minutes -= hours * 60;

    println!(
        "Estimated time remaining in training: {}h {}m {}s",
        ansi(hours, num_cs),
        ansi(minutes, num_cs),
        ansi(seconds, num_cs),
    );
}
