use bulletformat::ChessBoard;

use super::SparseInputType;

const PASSED_PAWN_MASKS: [u64; 64] = {
    // a pawn is passed if there are no enemy pawns in front of it on the same or adjacent files,
    // up to the end of the board.
    let mut masks = [0; 64];
    let mut i = 0;
    while i < 64 {
        let file = i % 8;
        let rank = i / 8;
        let mut mask = 0;
        // create in front:
        if rank < 7 {
            // same file
            mask |= 1 << (i + 8);
            // adjacent files
            if file > 0 {
                mask |= 1 << (i + 7);
            }
            if file < 7 {
                mask |= 1 << (i + 9);
            }
        }
        // fill in front:
        let mut r = rank + 1;
        while r < 8 {
            mask |= mask << 8;
            r += 1;
        }
        masks[i] = mask;
        i += 1;
    }
    masks
};

const DOUBLED_PAWN_MASKS: [u64; 64] = {
    // a pawn is doubled if there is another friendly pawn in front of it on the same file, up to the end of the board.
    let mut masks = [0; 64];
    let mut i = 0;
    while i < 64 {
        let rank = i / 8;
        let mut mask = 0;
        // in front:
        if rank < 7 {
            mask |= 1 << (i + 8);
        }
        // fill in front:
        let mut r = rank + 1;
        while r < 8 {
            mask |= mask << 8;
            r += 1;
        }
        masks[i] = mask;
        i += 1;
    }
    masks
};

#[derive(Clone, Copy, Debug, Default)]
pub struct Chess768Passers;
impl SparseInputType for Chess768Passers {
    type RequiredDataType = ChessBoard;

    /// The total number of inputs
    fn num_inputs(&self) -> usize {
        768
    }

    /// The maximum number of active inputs
    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        // construct stm & ntm pawn bitboards:
        let mut stm_pawns = 0;
        let mut ntm_pawns = 0;
        for (piece, square) in pos.into_iter() {
            let c = usize::from(piece & 8 > 0);
            let pc = usize::from(piece & 7);
            let sq = usize::from(square);
            if pc == 0 {
                if c == 0 {
                    stm_pawns |= 1 << sq;
                } else {
                    ntm_pawns |= 1 << sq;
                }
            }
        }
        for (piece, square) in pos.into_iter() {
            let c = usize::from(piece & 8 > 0);
            let pc = usize::from(piece & 7);
            let sq = usize::from(square);

            // if the pawn is passed & not doubled, add the passed pawn feature:
            if pc == 0 && c == 0 {
                let is_passed = (ntm_pawns & PASSED_PAWN_MASKS[sq]) == 0;
                let is_doubled = (stm_pawns & DOUBLED_PAWN_MASKS[sq]) != 0;
                if is_passed && !is_doubled {
                    // write feature as a back-rank pawn on the same file.
                    let file = sq % 8;
                    let rank = 0;
                    let passed_pawn_sq = rank * 8 + file;
                    let stm = [0, 384][c] + 64 * usize::from(piece & 7) + passed_pawn_sq;
                    let ntm = [384, 0][c] + 64 * usize::from(piece & 7) + (passed_pawn_sq ^ 56);
                    f(stm, ntm);
                }
            }
            // inverted for ntm pawns:
            if pc == 0 && c == 1 {
                // mirror the board vertically, so black pawns can be moving “up” the board.
                let is_passed = (stm_pawns.swap_bytes() & PASSED_PAWN_MASKS[sq ^ 56]) == 0;
                let is_doubled = (ntm_pawns.swap_bytes() & DOUBLED_PAWN_MASKS[sq ^ 56]) != 0;
                if is_passed && !is_doubled {
                    let file = sq % 8;
                    let rank = 7;
                    let passed_pawn_sq = rank * 8 + file;
                    let stm = [0, 384][c] + 64 * usize::from(piece & 7) + passed_pawn_sq;
                    let ntm = [384, 0][c] + 64 * usize::from(piece & 7) + (passed_pawn_sq ^ 56);
                    f(stm, ntm);
                }
            }

            let stm = [0, 384][c] + 64 * pc + sq;
            let ntm = [384, 0][c] + 64 * pc + (sq ^ 56);
            f(stm, ntm)
        }
    }

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String {
        "768".to_string()
    }

    /// Description of the input type
    fn description(&self) -> String {
        "Default psqt chess inputs".to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    const fn make_feature(c: usize, pc: usize, file: usize, rank: usize) -> usize {
        [0, 384][c] + 64 * pc + (rank * 8 + file)
    }

    #[test]
    fn feature_generation() {
        // we have this board:
        // k . . . . . . .
        // . . . . . . . .
        // . . p p . . . .
        // . . . . . . . .
        // . . . . . . . P
        // . . . . P P . P
        // . . . . . . . .
        // . . . . . . . K
        // k7/8/2pp4/8/7P/4PP1P/8/7K w - - 0 1
        // mirroring each other, there are four pawns, two white, two black.
        // one pawn of each pair is passed, while the other is not.
        // additionally, on the H-file, there are two white pawns,
        // but due to doubling only one of them is passed.
        // as such, we should get the following features:
        let manual_features = [
            // black pawn on c6:
            make_feature(1, 0, 2, 5),
            // black pawn on d6:
            make_feature(1, 0, 3, 5),
            // white pawn on e3:
            make_feature(0, 0, 4, 2),
            // white pawn on f3:
            make_feature(0, 0, 5, 2),
            // white passed pawn on f3:
            make_feature(0, 0, 5, 0),
            // black passed pawn on c6:
            make_feature(1, 0, 2, 7),
            // white king on h1:
            make_feature(0, 5, 7, 0),
            // black king on a8:
            make_feature(1, 5, 0, 7),
            // white pawn on h3:
            make_feature(0, 0, 7, 2),
            // white pawn on h4:
            make_feature(0, 0, 7, 3),
            // white passed pawn on h4:
            make_feature(0, 0, 7, 0),
        ];

        let board = ChessBoard::from_str("k7/8/2pp4/8/7P/4PP1P/8/7K w - - 0 1 | 0 | 0").unwrap();

        let mut mapped_stm_features = Vec::new();
        let mut mapped_ntm_features = Vec::new();
        Chess768Passers.map_features(&board, |stm, ntm| {
            mapped_stm_features.push(stm);
            mapped_ntm_features.push(ntm);
        });

        for feature in &manual_features {
            assert!(mapped_stm_features.contains(feature), "missing feature {}", feature);
        }

        for feature in &mapped_stm_features {
            assert!(manual_features.contains(feature), "unexpected feature {}", feature);
        }

        // assert no duplicates!
        let len_manual_features = manual_features.len();
        let len_mapped_stm_features = mapped_stm_features.len();
        let len_manual_features_set: usize = manual_features.iter().collect::<std::collections::HashSet<_>>().len();
        let len_mapped_stm_features_set: usize =
            mapped_stm_features.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(len_manual_features, len_manual_features_set, "manual features contain duplicates");
        assert_eq!(len_mapped_stm_features, len_mapped_stm_features_set, "mapped stm features contain duplicates");

        let inv_manual_features: Vec<_> = manual_features
            .iter()
            .map(|&feat| {
                let c = feat / 384;
                let pc = (feat % 384) / 64;
                let sq = feat % 64;
                [0, 384][c ^ 1] + 64 * pc + (sq ^ 56)
            })
            .collect();

        for feature in &mapped_ntm_features {
            assert!(inv_manual_features.contains(feature), "unexpected ntm feature {}", feature);
        }

        for feature in &inv_manual_features {
            assert!(mapped_ntm_features.contains(feature), "missing ntm feature {}", feature);
        }
    }
}
