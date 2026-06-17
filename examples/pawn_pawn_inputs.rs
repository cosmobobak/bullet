use bullet_lib::game::{formats::bulletformat::ChessBoard, inputs};

use montyformat::chess::{Piece, Side};

fn map_bb<F: FnMut(usize)>(mut bb: u64, mut f: F) {
    while bb > 0 {
        let sq = bb.trailing_zeros() as usize;
        f(sq);
        bb &= bb - 1;
    }
}

fn flip_horizontal(bb: u64) -> u64 {
    bb.swap_bytes().reverse_bits()
}

fn build_bbs(pos: &ChessBoard) -> [u64; 8] {
    let mut bbs = [0u64; 8];
    for (pc, sq) in pos.into_iter() {
        let pt = 2 + usize::from(pc & 7);
        let c = usize::from(pc & 8 > 0);
        let bit = 1 << sq;
        bbs[c] |= bit;
        bbs[pt] |= bit;
    }
    bbs
}

fn flip_view(mut bbs: [u64; 8]) -> [u64; 8] {
    bbs.swap(Side::WHITE, Side::BLACK);
    for bb in bbs.iter_mut() {
        *bb = bb.swap_bytes();
    }
    bbs
}

fn normalize_hm(mut bbs: [u64; 8]) -> [u64; 8] {
    let ksq = (bbs[Side::WHITE] & bbs[Piece::KING]).trailing_zeros();
    if ksq % 8 > 3 {
        for bb in bbs.iter_mut() {
            *bb = flip_horizontal(*bb);
        }
    }
    bbs
}

use crate::threat_inputs::ThreatInputs;

#[derive(Clone, Copy)]
pub struct PawnPawnInputs {
    threats: ThreatInputs,
    masks: [u64; 64],
}

impl PawnPawnInputs {
    pub const TOTAL_PAIRS: usize = 96 * 95 / 2;
    // pub const TOTAL_THREATS: usize = ThreatInputs::TOTAL_THREATS;
    const MAX_PAIRS: usize = 16 * 15 / 2;

    pub fn new(buckets: [usize; 32], masks: [u64; 64]) -> Self {
        Self { threats: ThreatInputs::new(buckets), masks }
    }

    fn pawn_id(colour: usize, sq: usize) -> usize {
        colour * 48 + sq - 8
    }

    fn pair_index(id_a: usize, id_b: usize) -> usize {
        let lo = id_a.min(id_b);
        let hi = id_a.max(id_b);
        hi * (hi - 1) / 2 + lo
    }

    fn collect_pairs(&self, bbs: [u64; 8]) -> ([(usize, usize); Self::MAX_PAIRS], usize) {
        let friendly = bbs[Side::WHITE] & bbs[Piece::PAWN];
        let enemy = bbs[Side::BLACK] & bbs[Piece::PAWN];

        let mut pairs = [(0usize, 0usize); Self::MAX_PAIRS];
        let mut n = 0;

        self.emit_same_colour(friendly, 0, &mut pairs, &mut n);
        self.emit_cross_colour(friendly, enemy, &mut pairs, &mut n);
        self.emit_same_colour(enemy, 1, &mut pairs, &mut n);

        (pairs, n)
    }

    fn emit_same_colour(&self, bb: u64, colour: usize, pairs: &mut [(usize, usize); Self::MAX_PAIRS], n: &mut usize) {
        let mut outer = bb;
        while outer != 0 {
            let sq_a = outer.trailing_zeros() as usize;
            outer &= outer - 1;
            let id_a = Self::pawn_id(colour, sq_a);
            map_bb(outer & self.masks[sq_a], |sq_b| {
                pairs[*n] = (id_a, Self::pawn_id(colour, sq_b));
                *n += 1;
            });
        }
    }

    fn emit_cross_colour(
        &self,
        friendly: u64,
        enemy: u64,
        pairs: &mut [(usize, usize); Self::MAX_PAIRS],
        n: &mut usize,
    ) {
        map_bb(friendly, |sq_a| {
            let id_a = Self::pawn_id(0, sq_a);
            map_bb(enemy & self.masks[sq_a], |sq_b| {
                pairs[*n] = (id_a, Self::pawn_id(1, sq_b));
                *n += 1;
            });
        });
    }
}

#[allow(dead_code)]
pub fn full_mask() -> [u64; 64] {
    [!0u64; 64]
}

pub fn three_file_band_mask() -> [u64; 64] {
    const A: u64 = 0x0101_0101_0101_0101;
    let mut masks = [0u64; 64];
    let mut sq = 8;
    while sq < 56 {
        let f = sq & 7;
        let mut m: u64 = A << f;
        if f > 0 {
            m |= A << (f - 1);
        }
        if f < 7 {
            m |= A << (f + 1);
        }
        masks[sq] = m;
        sq += 1;
    }
    masks
}

impl inputs::SparseInputType for PawnPawnInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        Self::TOTAL_PAIRS + self.threats.num_inputs()
    }

    fn max_active(&self) -> usize {
        self.threats.max_active() + Self::MAX_PAIRS
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        self.threats.map_features(pos, |stm, ntm| {
            f(Self::TOTAL_PAIRS + stm, Self::TOTAL_PAIRS + ntm);
        });

        let bbs = build_bbs(pos);
        let stm_bbs = normalize_hm(bbs);
        let ntm_bbs = normalize_hm(flip_view(bbs));

        let (stm_pairs, stm_count) = self.collect_pairs(stm_bbs);
        let (ntm_pairs, ntm_count) = self.collect_pairs(ntm_bbs);

        assert_eq!(stm_count, ntm_count);

        for i in 0..stm_count {
            let stm_idx = Self::pair_index(stm_pairs[i].0, stm_pairs[i].1);
            let ntm_idx = Self::pair_index(ntm_pairs[i].0, ntm_pairs[i].1);
            f(stm_idx, ntm_idx);
        }
    }

    fn shorthand(&self) -> String {
        todo!();
    }

    fn description(&self) -> String {
        todo!();
    }
}
