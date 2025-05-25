use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::PathBuf,
};

use rand::Rng;
use structopt::StructOpt;

#[derive(StructOpt)]
pub struct InterleaveOptions {
    #[structopt(required = true, min_values = 2)]
    pub inputs: Vec<PathBuf>,
    #[structopt(required = true, short, long)]
    pub output: PathBuf,
}

impl InterleaveOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        println!("Writing to {:#?}", self.output);
        println!("Reading from:\n{:#?}", self.inputs);
        let mut streams = Vec::new();
        let mut total = 0;

        let target = File::create(&self.output)?;
        // let validation_target = File::create(self.output.with_extension("validation"))?;
        let mut writer = BufWriter::new(target);
        // let mut validation_writer = BufWriter::new(validation_target);

        let mut total_input_file_size = 0;
        for path in &self.inputs {
            let file = File::open(path)?;

            let count = file.metadata()?.len();

            total_input_file_size += count;

            if count > 0 {
                let fname =
                    path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "<unknown>".into());
                streams.push((count, BufReader::new(file), fname));
                total += count;
            }
        }

        let mut remaining = total;
        let mut rng = rand::rng();

        const INTERVAL: u64 = 1024 * 1024 * 256;
        let mut prev = remaining / INTERVAL;

        let mut buffer = Vec::new();
        let mut games = 0usize;

        while remaining > 0 {
            let mut spot = rng.random_range(..remaining);
            let mut idx = 0;
            while streams[idx].0 < spot {
                spot -= streams[idx].0;
                idx += 1;
            }

            let (count, reader, _) = &mut streams[idx];

            buffer.clear();

            let mut initial_position = [0; 32];
            reader.read_exact(&mut initial_position)?;
            buffer.extend_from_slice(&initial_position);
            loop {
                let mut buf = [0; 4];
                reader.read_exact(&mut buf)?;
                buffer.extend_from_slice(&buf);
                if buf == [0; 4] {
                    break;
                }
            }

            writer.write_all(&buffer)?;
            games += 1;

            let size = buffer.len() as u64;

            remaining -= size;
            *count -= size;
            if *count == 0 {
                println!("Finished reading {}", streams[idx].2);
                streams.swap_remove(idx);
            }

            if remaining / INTERVAL < prev {
                prev = remaining / INTERVAL;
                let written = total - remaining;
                print!("Written {written}/{total} Bytes ({:.2}%)\r", written as f64 / total as f64 * 100.0);
                let _ = std::io::stdout().flush();
            }
        }

        writer.flush()?;

        println!();
        println!("Written {games} games to {:#?}", self.output);

        let output_file = File::open(&self.output)?;
        let output_file_size = output_file.metadata()?.len();
        if output_file_size != total_input_file_size {
            anyhow::bail!("Output file size {output_file_size} does not match input file size {total_input_file_size}");
        }

        Ok(())
    }
}
