# Dark Fermion Model

A repository for all code related to the dark fermion model.

To process simulated or real data (previously run through the CutFlow analyzer), stored in a `ROOT` file, instantiate the class `processData`, passing the dataset type:

  1. `'mc'`: Monte-Carlo (MC) simulated data samples, containing both the `sel`-level (generator level, e.g., the hard-scattering process) information, and the reco-level (reconstructed, or `sel`) data.
  2. `'bkg'`: MC simulated samples of specific background processes with four final-state muons. Only `sel` level information is present.
  3. `'sig'`: Signal samples (MC or real data) which only contain `sel` level information.
  4. `'ntuple'`: Bare n-tuples (not passed through the CutFlow analyzer) which only contains `sel` level information.
  
  Below, we list the usage of the various classes, with test cases corresponding to the dataset types 1–4 above.

## Processing root files
To process `ROOT` files, we instantiate the `processData` class. For case 1 (`'mc'` samples):

```
data = processData('mc')
```

To extract data, we use the `extract_data()` method:

```
data.extract_data('path/to/your/root_file.root', 'root_directory')
```

To apply preliminary cuts (to remove the improperly reconstructed events with pT, eta, phi, and charge), execute the `prelim_cuts()` method:

```
data.prelim_cuts()
```

For case 1 only, we calculate the difference in delta R (dR = sqrt{dEta^2 + dPhi^2}) between the GEN level and the SEL level muons, creating a 4 x 4 array for each event. This is used for determining if the muons were reconstructed properly.

```
data.dR_gen_calc()
```

The use the stochastic sampling method (SSM) to pseudorandomly sample the muons and then determinine the minimum delta R between the GEN and SEL level muons (calculated using `dR_gen_calc()`). Outline of procedure:

for each event:
  1. Retrieve both the sel and gen level charge information of the muons.
  2. Pseudorandomly select a muon [0, 3].
  3. Based off of this muon, determine the oppositely charged muons and the other muon with same charge.
  4. Calculate the minimum dR (dR = sqrt{eta^2 + phi^2}; previously calculated) between the gen and sel muons for same and opposite charge muons.
      1. If pseudorandomly chosen muon has minimum dR of the two same/opposite charged muons: Label that muon and remove the index from the total list of charge.
      2. Pair the other sel muon with the other same charge gen muon.
  5. From the minima previously calculated, determine which opposite charged muon has the minimum dR.
      1. Pair the remaining oppositely charged sel muon to the last muon available.
      
```
data.ssm()
```

To apply the cut on the dR *__between the GEN and SEL muons__* in each dimuon:

```
data.dR_cut()
```

For machine learning (ML) algorithm training purposes, we generate the incorrect permutations by considering the charge and order of the muons. To generate these permutation, execute

```
data.permutations()
```

Now, to calculate the invariant mass, we call

```
data.inv_mass_calc()
```

For the final, "official" slection of the four final-state muons for each event, we calculate the dR and dphi between the muons in each dimuon pair by executing:

```
data.dR_diMu()
```

To calculate the event shape variables, run

```
data.event_shapes()
```

To get the final result in `.csv` format, we execute:

```
final_array = data.fill_and_sort(save=True)
```

This procedure is very similar for the other MC/real data signal-like dataset types. For example, the full command chain to process the signal (`'sig'`) datasets and the n-tuple (`'ntuple'`) datasets contains all of the same commands *except* the commands where the dR between the GEN and SEL level muons are computed, or where cuts are made on GEN level information. (This is because GEN level information is not present in the `ROOT` file.) For either the 'sig'` or `'ntuple'`  datasets, the following chain is appropriate:

```
data = processData('sig') # or 'ntuple'
data.prelim_cuts()
data.match_bkg_mu()
data.dR_cut(dRcut, cut)
data.permutations()
data.inv_mass_calc()
data.dR_diMu()
final_array = data.fill_and_sort(save = True)
```

The main difference in processing ROOT filelies in the `bkg` (background) datasets




  
