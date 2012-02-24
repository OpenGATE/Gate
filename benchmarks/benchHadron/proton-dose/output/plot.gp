
reset
set term wxt
shift=4.1

set title "New Gate, G4_WATER"

## according to 140 MeV
normalisation_measure_100 = 127.6
normalisation_measure_140 = 120.6
normalisation_measure_180 = 111.7
normalisation_measure_226 = 106.2

normalisation_simu_100 = `! perl -e '$max=-1e38; while (<>) {@t=split; $max=$t[0] if $t[0]>$max}; print $max' < PDD-Chamber-100-Mev-Dose.txt`
normalisation_simu_140 = `! perl -e '$max=-1e38; while (<>) {@t=split; $max=$t[0] if $t[0]>$max}; print $max' < PDD-Chamber-140-Mev-Dose.txt`
normalisation_simu_180 = `! perl -e '$max=-1e38; while (<>) {@t=split; $max=$t[0] if $t[0]>$max}; print $max' < PDD-Chamber-180-Mev-Dose.txt`
normalisation_simu_226 = `! perl -e '$max=-1e38; while (<>) {@t=split; $max=$t[0] if $t[0]>$max}; print $max' < PDD-Chamber-226.7-Mev-Dose.txt`

plot [0:350] "../data/BP-100.txt" using ($1+shift):($2/normalisation_measure_100) title "Measure 100 MeV" w l
replot "../data/BP-140.txt" using ($1+shift):($2/normalisation_measure_140) title "Measure 140 Mev" w l
replot "../data/BP-180.txt" using ($1+shift):($2/normalisation_measure_180) title "Measure 180 Mev" w l
replot "../data/BP-226.7.txt" using ($1+shift):($2/normalisation_measure_226) title "Measure 226.7 Mev" w l

replot "PDD-Chamber-100-Mev-Dose.txt" using ($0/2):($1/normalisation_simu_100) title "Simu (chamber) 100 Mev" axes x1y1 w l
replot "PDD-Chamber-140-Mev-Dose.txt" using ($0/2):($1/normalisation_simu_140) title "Simu (chamber) 140 Mev" axes x1y1 w l
replot "PDD-Chamber-180-Mev-Dose.txt" using ($0/2):($1/normalisation_simu_180) title "Simu (chamber) 180 Mev" axes x1y1 w l
replot "PDD-Chamber-226.7-Mev-Dose.txt" using ($0/2):($1/normalisation_simu_226) title "Simu (chamber) 226.7 Mev" axes x1y1 w l

# normalisation_simu_I_100 = `! perl -e '$max=-1e38; while (<>) {@t=split; $max=$t[0] if $t[0]>$max}; print $max' < PDD-Integral-100-Mev-Dose.txt`
# normalisation_simu_I_140 = `! perl -e '$max=-1e38; while (<>) {@t=split; $max=$t[0] if $t[0]>$max}; print $max' < PDD-Integral-140-Mev-Dose.txt`
# normalisation_simu_I_180 = `! perl -e '$max=-1e38; while (<>) {@t=split; $max=$t[0] if $t[0]>$max}; print $max' < PDD-Integral-180-Mev-Dose.txt`
# normalisation_simu_I_226 = `! perl -e '$max=-1e38; while (<>) {@t=split; $max=$t[0] if $t[0]>$max}; print $max' < PDD-Integral-226.7-Mev-Dose.txt`

# replot "PDD-Integral-100-Mev-Dose.txt" using ($0/2):($1/normalisation_simu_I_100) title "Simu (integral) 100 Mev" axes x1y1 w p
# replot "PDD-Integral-140-Mev-Dose.txt" using ($0/2):($1/normalisation_simu_I_140) title "Simu (integral) 140 Mev" axes x1y1 w p
# replot "PDD-Integral-180-Mev-Dose.txt" using ($0/2):($1/normalisation_simu_I_180) title "Simu (integral) 180 Mev" axes x1y1 w p
# replot "PDD-Integral-226.7-Mev-Dose.txt" using ($0/2):($1/normalisation_simu_I_226) title "Simu (integral) 226.7 Mev" axes x1y1 w p

set term post
set output "a.ps"
replot
set term x11
