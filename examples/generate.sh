#!/usr/bin/env bash
# Generate all example outputs from sample structures.

set -euo pipefail

OUT=$(dirname $(realpath ${BASH_SOURCE[0]%/*}))
DIR=$OUT/structures
mkdir -p "$OUT"

echo "=== Presets ==="
xyzrender "$DIR/caffeine.xyz" -o "$OUT/caffeine_default.svg"
xyzrender "$DIR/caffeine.xyz" -o "$OUT/caffeine_default.png"
xyzrender "$DIR/caffeine.xyz" --config flat -o "$OUT/caffeine_flat.svg"
xyzrender "$DIR/caffeine.xyz" --config paton -o "$OUT/caffeine_paton.svg"

echo "=== Display options ==="
xyzrender "$DIR/ethanol.xyz" --hy -o "$OUT/ethanol_all_h.svg"           # all H
xyzrender "$DIR/ethanol.xyz" --hy 7 8 9 -o "$OUT/ethanol_some_h.svg"   # specific H atoms
xyzrender "$DIR/ethanol.xyz" --no-hy -o "$OUT/ethanol_no_h.svg"        # no H
xyzrender "$DIR/benzene.xyz" --hy -o "$OUT/benzene.svg"                 # aromatic
xyzrender "$DIR/caffeine.xyz" --bo -k -o "$OUT/caffeine_kekule.svg"    # Kekule bond orders

echo "=== VdW spheres ==="
xyzrender "$DIR/asparagine.xyz" --hy --vdw -o "$OUT/asparagine_vdw.svg"  # all atoms
xyzrender "$DIR/asparagine.xyz" --hy --vdw "1-6" -o "$OUT/asparagine_vdw_partial.svg"  # some atoms
xyzrender "$DIR/asparagine.xyz" --hy --vdw --config paton -o "$OUT/asparagine_vdw_paton.svg"  # all atoms

echo "=== QM output files ==="
xyzrender "$DIR/bimp.out" -o "$OUT/bimp_qm.svg" 
xyzrender "$DIR/mn-h2.log" -o "$OUT/mn-h2_qm.svg" --ts

echo "=== Input files ==="
xyzrender "$DIR/ala_phe_ala.pdb" -o "$OUT/ala_phe_ala.svg"
xyzrender --smi "C1CCCCC1" --hy -o "$OUT/cyclohexane_smi.svg"

echo "=== TS and NCI options ==="
if [ -n "$(which ORCA)" ]; then
    xyzrender "$DIR/sn2.out" --ts-bond "1-2" -o "$OUT/sn2_ts_man.svg"
    xyzrender "$DIR/sn2.out" --ts --hy -o "$OUT/sn2_ts.svg" 
    xyzrender "$DIR/bimp.out" --nci -o "$OUT/bimp_nci.svg"  # all NCI bonds
else
    echo "ORCA not found, skipping some examples."
fi
xyzrender "$DIR/Hbond.xyz" --hy --nci-bond "8-9" -o "$OUT/nci_man.svg"  # specific NCI bond only
xyzrender "$DIR/Hbond.xyz" --hy --nci -o "$OUT/nci.svg"  # specific NCI bond only


echo "=== Annotations & measurements ==="
xyzrender "$DIR/caffeine.xyz" --idx -o "$OUT/caffeine_idx.svg" 
xyzrender "$DIR/caffeine.xyz" --idx n --hy --label-size 25 -o "$OUT/caffeine_idx_n.svg" 
xyzrender "$DIR/caffeine.xyz" --hy --cmap "$DIR/caffeine_charges.txt" -o "$OUT/caffeine_cmap.svg" --gif-rot -go "$OUT/caffeine_cmap.gif"
xyzrender "$DIR/caffeine.xyz" --hy --cmap "$DIR/caffeine_charges.txt" -o "$OUT/caffeine_cmap.svg" --cmap-range -0.5 0.5
xyzrender "$DIR/caffeine.xyz" -l 13 6 9 4 t -l 1 a -l 14 d -l 7 12 8 a -l 11 d -o "$OUT/caffeine_dihedral.svg"
xyzrender "$DIR/caffeine.xyz" -l 1 best -l 2 "NBO: 0.4" -o "$OUT/caffeine_labels.svg"
if [ -n "$(which ORCA)" ]; then
    xyzrender "$DIR/sn2.out" --ts --label "$OUT/sn2_label.txt" -o "$OUT/sn2_ts_label.svg" --label-size 40
else
    echo "ORCA not found, skipping some examples."
fi

echo "=== Molecular orbitals ==="
xyzrender "$DIR/caffeine_lumo.cube" --mo --mo-colors maroon teal -o "$OUT/caffeine_lumo.svg"
xyzrender "$DIR/caffeine_homo.cube" --mo --hy --iso 0.03 -o "$OUT/caffeine_homo_iso_hy.svg"
xyzrender "$DIR/caffeine_homo.cube" --mo -o "$OUT/caffeine_homo_rot.svg" --gif-rot -go "$OUT/caffeine_homo.gif"

echo "=== Density surface ==="
xyzrender "$DIR/caffeine_dens.cube" --dens --iso 0.01 -o "$OUT/caffeine_dens_iso.svg"
xyzrender "$DIR/caffeine_dens.cube" --dens --dens-color teal --opacity 0.75 -o "$OUT/caffeine_dens_custom.svg"
xyzrender "$DIR/caffeine_dens.cube" --dens -o "$OUT/caffeine_dens.svg" --gif-rot -go "$OUT/caffeine_dens.gif"

echo "=== ESP surface ==="
xyzrender "$DIR/caffeine_dens.cube" --esp "$DIR/caffeine_esp.cube" -o "$OUT/caffeine_esp.svg"
xyzrender "$DIR/caffeine_dens.cube" --esp "$DIR/caffeine_esp.cube" --iso 0.005 --opacity 0.75 -o "$OUT/caffeine_esp_custom.svg"

echo "=== GIF animations ==="
xyzrender "$DIR/caffeine.xyz" -o "$OUT/caffeine_gif.svg" --gif-rot -go "$OUT/caffeine.gif"
xyzrender "$DIR/caffeine.xyz" -o "$OUT/caffeine_xy.svg" --gif-rot xy -go "$OUT/caffeine_xy.gif"
xyzrender "$DIR/bimp.out" -o "$OUT/bimp_rot.svg" --gif-rot --gif-ts --vdw 84-169 -go "$OUT/bimp.gif"
xyzrender "$DIR/bimp.out" -o "$OUT/bimp_trj.svg" --gif-trj --ts -go "$OUT/bimp_trj.gif"
xyzrender "$DIR/mn-h2.log" -o "$OUT/mn-h2_gif.svg" --gif-ts -go "$OUT/mn-h2.gif"
xyzrender "$DIR/bimp.out" -o "$OUT/bimp_nci.svg" --ts --gif-trj --vdw 84-169 --nci -go "$OUT/bimp_nci_trj.gif"
xyzrender "$DIR/bimp.out" -o "$OUT/bimp_nci.svg" --gif-ts --gif-rot --vdw 84-169 --nci -go "$OUT/bimp_nci_ts.gif"

echo "=== Crystal / unit cell ==="
xyzrender "$DIR/caffeine_cell.xyz" --cell -o "$OUT/caffeine_cell.svg" --no-orient --gif-rot -go "$OUT/caffeine_cell.gif" 

echo "=== Crystal / periodic structures ==="
xyzrender "$DIR/NV63.vasp" --crystal -o "$OUT/NV63_vasp.svg" --gif-rot -go "$OUT/NV63_vasp.gif"  # auto-detected as VASP
xyzrender "$DIR/NV63.in" --crystal qe -o "$OUT/NV63_qe.svg"          # explicit QE mode
xyzrender "$DIR/NV63.vasp" --crystal --no-ghosts -o "$OUT/NV63_vasp_no_img.svg"       # hide image atoms
xyzrender "$DIR/NV63.vasp" --crystal --no-cell -o "$OUT/NV63_vasp_no_cell.svg"        # hide unit cell box
xyzrender "$DIR/NV63.vasp" --crystal --axis 111 --gif-rot 111 -o "$OUT/NV63_vasp_111.svg" -go "$OUT/NV63_vasp_111.gif"  # look down [111], rotate around [111]

echo "Done! Outputs written to $OUT/"
