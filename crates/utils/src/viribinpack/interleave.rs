use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
};

use anyhow::Context;
use structopt::StructOpt;
use viriformat::dataformat::Game;

use crate::Rand;

#[derive(StructOpt)]
pub struct InterleaveOptions {
    #[structopt(required = true, min_values = 2)]
    pub inputs: Vec<PathBuf>,
    #[structopt(required = true, short, long)]
    pub output: PathBuf,
}

impl InterleaveOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        println!("Writing to {}", self.output.display());
        println!("Reading from:");
        for path in &self.inputs {
            println!("  {}", path.display());
        }
        let mut streams = Vec::new();
        let mut total = 0;

        let target = File::create(&self.output)
            .with_context(|| format!("Failed to create output file {}", self.output.display()))?;
        let mut writer = BufWriter::new(target);

        let mut total_input_file_size = 0;
        for path in &self.inputs {
            let file = File::open(path).with_context(|| format!("Failed to open input file {}", path.display()))?;

            let metadata =
                file.metadata().with_context(|| format!("Failed to get metadata for input file {}", path.display()))?;
            let count = metadata.len();

            total_input_file_size += count;

            if count > 0 {
                let fname =
                    path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "<unknown>".into());
                streams.push((count, BufReader::new(file), fname));
                total += count;
            }
        }

        let mut remaining = total;
        let mut rng = Rand::default();

        const INTERVAL: u64 = 1024 * 1024 * 256;
        let mut prev = remaining / INTERVAL;

        let mut buffer = Vec::new();
        let mut games = 0usize;

        while remaining > 0 {
            let mut spot = rng.rand() % remaining;
            let mut idx = 0;
            while streams[idx].0 < spot {
                spot -= streams[idx].0;
                idx += 1;
            }

            let (count, reader, name) = &mut streams[idx];

            buffer.clear();
            Game::deserialise_fast_into_buffer(reader, &mut buffer)
                .with_context(|| format!("Failed to read game from {name}"))?;
            writer.write_all(&buffer).with_context(|| format!("Failed to write game from {name}"))?;
            games += 1;

            let size = buffer.len() as u64;

            remaining -= size;
            *count -= size;
            if *count == 0 {
                println!("Finished reading {name}");
                streams.swap_remove(idx);
            }

            if remaining / INTERVAL < prev {
                prev = remaining / INTERVAL;
                let written = total - remaining;
                print!("Written {written}/{total} Bytes ({:.2}%)\r", written as f64 / total as f64 * 100.0);
                let _ = std::io::stdout().flush();
            }
        }

        writer.flush().with_context(|| format!("Failed to flush output file {}", self.output.display()))?;

        println!();
        println!("Written {games} games to {}", self.output.display());

        let output_file = File::open(&self.output)
            .with_context(|| format!("Failed to open output file {}", self.output.display()))?;
        let metadata = output_file
            .metadata()
            .with_context(|| format!("Failed to get metadata for output file {}", self.output.display()))?;
        let output_file_size = metadata.len();
        if output_file_size != total_input_file_size {
            anyhow::bail!("Output file size {output_file_size} does not match input file size {total_input_file_size}");
        }

        Ok(())
    }
}
