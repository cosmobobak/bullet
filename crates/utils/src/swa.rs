use std::{
    fs::{self, File},
    io::Write,
    path::PathBuf,
};

use anyhow::{Context, ensure};
use structopt::StructOpt;

/// SWA over checkpoints: averages the weights of ≥2 `raw.bin`s
/// and writes the result as a new checkpoint directory.
///
/// Only works for checkpoints saved with plain float `SavedFormat`s.
#[derive(StructOpt)]
pub struct SwaOptions {
    /// checkpoint directories (containing raw.bin), or raw.bin paths directly
    #[structopt(required = true, min_values = 2)]
    pub inputs: Vec<PathBuf>,
    /// output checkpoint directory
    #[structopt(required = true, short, long)]
    pub output: PathBuf,
}

impl SwaOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        #![expect(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

        let paths: Vec<PathBuf> =
            self.inputs.iter().map(|p| if p.is_dir() { p.join("raw.bin") } else { p.clone() }).collect();

        println!("Averaging {} checkpoints:", paths.len());

        let mut acc: Vec<f64> = Vec::new();

        for path in &paths {
            let bytes = fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
            ensure!(bytes.len() % 4 == 0, "{} is not a whole number of f32s", path.display());

            if acc.is_empty() {
                acc = vec![0.0; bytes.len() / 4];
            }

            ensure!(
                bytes.len() / 4 == acc.len(),
                "{} does not match the size of preceding checkpoints",
                path.display()
            );

            for (a, b) in acc.iter_mut().zip(bytes.chunks_exact(4)) {
                *a += f64::from(f32::from_le_bytes(b.try_into().unwrap()));
            }

            println!("  {}", path.to_string_lossy());
        }

        let scale = 1.0 / paths.len() as f64;
        let mut buf = Vec::with_capacity(acc.len() * 4);
        for a in &acc {
            buf.extend_from_slice(&((a * scale) as f32).to_le_bytes());
        }

        fs::create_dir_all(&self.output).with_context(|| "Failed to create output directory")?;

        File::create(self.output.join("raw.bin"))?.write_all(&buf)?;

        // quantised.bin for float save formats
        let overhang = buf.len() % 64;
        if overhang > 0 {
            let chs = [b'b', b'u', b'l', b'l', b'e', b't'];
            for i in 0..64 - overhang {
                buf.push(chs[i % chs.len()]);
            }
        }
        File::create(self.output.join("quantised.bin"))?.write_all(&buf)?;

        println!("Wrote {} f32s to {}", acc.len(), self.output.display());

        Ok(())
    }
}
