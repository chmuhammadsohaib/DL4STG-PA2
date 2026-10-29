# Pre-registered predictions (written before the corresponding experiments)

## Q2.5 (before Testing the Baseline)

Mirror: model A should roughly pass. Moving MNIST draws directions uniformly, so mirrored trajectories are already familiar motion; only the digit shapes are new. I expect C_M to be small relative to E_M and E_M only slightly above E^(5), with the main damage being blurrier or slightly distorted digits rather than wrong motion.

Rotation: also roughly passes for the same reason (a 90 degree rotation of a uniform direction distribution is still uniform), but somewhat worse than mirror, because rotated digits look less like any training digit than mirrored ones (e.g. a rotated 1 is a horizontal bar). I expect C_R > C_M and E_R > E_M, by a modest amount.

Speed: model A's reach is 2 folded cells = 8 original pixels per step. Typical Moving MNIST speed should be about 3-4 px/frame, so 2x (about 7 px/frame) is just inside the reach but never seen in training, and 3x (about 10-11 px/frame) exceeds the reach. I predict a clear degradation at 2x and a breakdown at 3x (digits stall, blur out or vanish), with 3x E^(2) close to or worse than Copy-last.

## Q2.11 (before training B-D)

Model B (symmetry augmentation) should shrink C_M and C_R and E_M, E_R to about E^(5), fixing the appearance part of the mirror/rotation failures, but not change the speed results. Model C (speed augmentation) should fix 2x, because 2x motion is still within the 8 px reach and only unfamiliar, but should still fail at 3x because that motion is beyond the reach. Model D (larger reach plus speed augmentation) should fix 3x as well, with similar or slightly worse 1x error and no gain on mirror/rotation consistency.
