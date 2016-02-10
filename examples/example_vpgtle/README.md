## vpgTLE example

vpgTLE is broken up into three parts. Stage 0 is required to be run once, and each vp
gTLE simulation is then broken up into Stage 1 and Stage 2. For each stage, you can find and example in the `examples/vpgTLE` directory.

To understand the background, physics and mathematics of this example, refer to `Accelerated Prompt Gamma estimation for clinical Proton Therapy simulations` by B.F.B. Huisman.

### Stage 0

Creating the PGdb means computing $\frac{\Gamma_Z}{\rho_Z}$ for the elements that appear in your simulations. In `stage0/` you can find all that is required to produce a pgdb. The runall-elements.sh script would create db-files for all the elements found in Schneiders conversion tables. The runall-elements-cluster.sh can be used together with `gate_run_submit_cluster.sh` to produce a pgdb with $10^9$ statistics per element.

Note that mac/main.mac can be used to alter the number of primaries per run, and the maximum proton energy which has been set at 200MeV. If your simulation in Stage 1 uses higher proton energies, the pgdb must be recomputed.

A pgdb.root with $10^9$ primaries per material has been supplied in the example for Stage 1.

### Stage 1

We take the head and neck phantom from the protontherapy example and produce the PGyd image. A prepared pgdb.root is found in the `data/` dir in `stage1/`. By default, the same plan is used as in the paper, but selected spot and the original plan have been supplied, also in `data/`.

One can produce PG output in three ways:

1. with the PromptGammaTLEActor. By default the actor is enabled (`mac/actors.tle.mac`).
2. with the PromptGammaAnalogActor. Same output as vpgTLE, but only analogically produced PGs are scored. Enable `mac/actors.analog.voxel.mac`.
3. with the PhaseSpaceActor. Also only scores analogically produced PGs.

The number of primaries required is a user choice. We recommend at least $10^6$ primaries for a sufficiently converged output. For visualization however, $10^3$ is enough (takes less than a minute).

Another important consideration is the region where the actors score. It is set to a region containing the layer of the treatment plan.

Run the simulation with `Gate mac/main.mac -a '[NPRIM,1000]'` to get the PGyd for $10^3$ primaries.

#### Stage 1 Debug outputs

Setting `/gate/actor/pgtle/enableDebugOutput` to true (uncomment them in the example) will give the user a bunch of debug output. It will take roughly 10 times the memory (disk and RAM). It will output all the GammaM and NgammaM tables to disk (TH2Ds, as function of proton and gamma energies). It will score the tracklengths and tracklengths squared explicitly, so the user could compute the vpgTLE output offline with the outputs so far. To show this is possible, the flag will also enable a second vpgTLE output, which should be identical to the regular vpgTLE output. A variance (random+systematic) per voxel image will also be outputted. If the user wants the variance over some projection, he should use the GammaM/NgammaM tables together with the tracklengths (the code of the PromptGammaTLEActor may be of help).
	
Secondly, `/gate/actor/pgtle/enableOutputMatch` will change the regular actor to use the material of the center of the voxel instead of the interaction point. This ensures, in the case that vpgTLE actor voxels and CT image voxels do not overlap, the output is bit for bit identical.

### Stage 2

To produce and propagate the PGs throughout the scene (into your PG detector), we execute Stage 2. Using the output of Stage 1 as a source, we use a toy PG collimated camera to record some signal. Again execute with `Gate mac/main.mac -a '[NPRIM,1000]'` where the number of primaries is now the number of protons for which you want to get a representative PG signal.

Generate an output `source.mhd` and `source.raw` in Stage 1 and move the files to `stage2/data` to make the example work. It outputs the gammas produced along the beam direction and a PhaseSpace of the PGs that make it through the collimator.