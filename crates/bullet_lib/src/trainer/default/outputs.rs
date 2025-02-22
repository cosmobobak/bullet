use bulletformat::ChessBoard;

pub trait OutputBuckets<T>: Send + Sync + Copy + Default + 'static {
    const BUCKETS: usize;

    fn bucket(&mut self, pos: &T) -> u8;
}

#[derive(Clone, Copy, Default)]
pub struct Single;
impl<T: 'static> OutputBuckets<T> for Single {
    const BUCKETS: usize = 1;

    fn bucket(&mut self, _: &T) -> u8 {
        0
    }
}

#[derive(Clone, Copy, Default)]
pub struct MaterialCount<const N: usize>;
impl<const N: usize> OutputBuckets<ChessBoard> for MaterialCount<N> {
    const BUCKETS: usize = N;

    fn bucket(&mut self, pos: &ChessBoard) -> u8 {
        let divisor = 32usize.div_ceil(N);
        (pos.occ().count_ones() as u8 - 2) / divisor as u8
    }
}

#[derive(Clone, Copy, Default)]
pub struct MaterialCountFarseer;
impl OutputBuckets<ChessBoard> for MaterialCountFarseer {
    const BUCKETS: usize = 8;

    fn bucket(&mut self, pos: &ChessBoard) -> u8 {
        const TABLE: [u8; 33] = [
            0,
            0, 0, 0, 0, 0, 0, // 1, 2, 3, 4, 5, 6
            0, 0, 0, 0, // 7, 8, 9, 10
            1, 1, 1,
            2, 2, 2,
            3, 3, 3,
            4, 4, 4,
            5, 5, 5,
            6, 6, 6,
            7, 7, 7, 7
        ];
        // const int bucket = bucketNo[pos.count<ALL_PIECES>()];
        let index = pos.occ().count_ones() as usize;
        TABLE[index.min(32)]
    }
}