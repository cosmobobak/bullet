use super::indices;

pub const PAWN: usize = 0;
pub const KNIGHT: usize = PAWN + 6 * indices::PAWN;
pub const BISHOP: usize = KNIGHT + 10 * indices::KNIGHT[64];
pub const ROOK: usize = BISHOP + 8 * indices::BISHOP[64];
pub const QUEEN: usize = ROOK + 8 * indices::ROOK[64];
pub const END: usize = QUEEN + 10 * indices::QUEEN[64];
