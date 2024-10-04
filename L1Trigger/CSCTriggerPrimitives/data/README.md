# L1Trigger-CSCTriggerPrimitives

General note: `.txt` files are ASCII format. The `.mem` files are transated `.txt` into hexadecimal and can be loaded into VERILOG firmware.

## CCLUT (under directory CCLUT)

This repository holds lookup-tables which map a pattern number and a 12-bit comparator code onto an unsigned 4-bit position offset and a 5-bit slope. In the current form, there are 5 patterns (0 through 4) and 4096 comparator codes. LUTs are also available with floating point values (for reference). A third set of LUTs convert the Run-1/2 patterns to Run-3 patterns.

The firmware dataword for each pattern and comparator code is 18 bits long:
   - [8:0] is quality (set all to 0 for now)
   - [12:9] is slope value
   - [13] is slope sign
   - [17:14] is position offset

## GEM-CSC LUTs (under directory GEMCSC)

This repository also holds lookup-tables that map GEM readout channels (pad) onto 1/8-strip CSC channels, and that map GEM roll numbers onto CSC wiregroups. LUTs are provided for ME1/1 and ME2/1, for even and odd. In the case of ME1/1, separate LUTs are present for ME1/a and ME1/b strips. A single LUT is foreseen for the wiregroups. The .mem files are those which will be loaded into VERILOG firmware.

* map GEM pad onto CSC 1/8-strip number
   - GEMCSCLUT_pad_es_ME1a_even.txt (start at CSC strip 64, or 1/8-strip 512)
   - GEMCSCLUT_pad_es_ME1a_odd.txt  (start at CSC strip 64, or 1/8-strip 512)
   - GEMCSCLUT_pad_es_ME1b_even.txt
   - GEMCSCLUT_pad_es_ME1b_odd.txt
   - GEMCSCLUT_pad_es_ME21_even.txt
   - GEMCSCLUT_pad_es_ME21_odd.txt
   - GEMCSCLUT_pad_hs_ME1a_even.txt
   - GEMCSCLUT_pad_hs_ME1a_odd.txt
   - GEMCSCLUT_pad_hs_ME1b_even.txt
   - GEMCSCLUT_pad_hs_ME1b_odd.txt
   - GEMCSCLUT_pad_hs_ME21_even.txt
   - GEMCSCLUT_pad_hs_ME21_odd.txt

* map GEM roll onto CSC minimum and maximum wiregroup number
   - GEMCSCLUT_roll_l1_max_wg_ME11_even.txt
   - GEMCSCLUT_roll_l1_max_wg_ME11_odd.txt
   - GEMCSCLUT_roll_l1_max_wg_ME21_even.txt
   - GEMCSCLUT_roll_l1_max_wg_ME21_odd.txt
   - GEMCSCLUT_roll_l1_min_wg_ME11_even.txt
   - GEMCSCLUT_roll_l1_min_wg_ME11_odd.txt
   - GEMCSCLUT_roll_l1_min_wg_ME21_even.txt
   - GEMCSCLUT_roll_l1_min_wg_ME21_odd.txt
   - GEMCSCLUT_roll_l2_max_wg_ME11_even.txt
   - GEMCSCLUT_roll_l2_max_wg_ME11_odd.txt
   - GEMCSCLUT_roll_l2_max_wg_ME21_even.txt
   - GEMCSCLUT_roll_l2_max_wg_ME21_odd.txt
   - GEMCSCLUT_roll_l2_min_wg_ME11_even.txt
   - GEMCSCLUT_roll_l2_min_wg_ME11_odd.txt
   - GEMCSCLUT_roll_l2_min_wg_ME21_even.txt
   - GEMCSCLUT_roll_l2_min_wg_ME21_odd.txt

## CSC ME1/1 LUTs (under directory ME11)

LUT for which ME1/1 wire group can cross which halfstrip. 1st index: WG number. 2nd index: inclusive HS range. -1 means no overlap

* map wiregroup onto min and max half-strip number that it crosses in ME1/a. Keep in mind that ME1A is considered an extension of ME1B. This means that ME1A half-strips start at 128 and end at 223
   - CSCLUT_wg_min_hs_ME1a.txt
   - CSCLUT_wg_max_hs_ME1a.txt

* map wiregroup onto min and max half-strip number that it crosses in ME1/a. When the half-strips are triple-ganged, (Run-1) ME1A half-strips go from 128 to 159
   - CSCLUT_wg_min_hs_ME1a_ganged.txt
   - CSCLUT_wg_max_hs_ME1a_ganged.txt

* map wiregroup onto min and max half-strip number that it crosses in ME1/b. ME1B half-strips start at 0 and end at 127
   - CSCLUT_wg_min_hs_ME1b.txt
   - CSCLUT_wg_max_hs_ME1b.txt

## LCT combination codes (under directory LCTCode)

* LUT that defines correspondence between ALCT-CLCT combination code and the resulting best and second lct
   - CSCLUT_code_to_bestLCT.txt
   - CSCLUT_code_to_secondLCT.txt

## GEM-CSC Slope correction

Repository for current lookup tables in decimal and hexadecimal format for the GEM-CSC slope correction.

.txt files contain absolute slope value corrections in 1/8th strip units from 0 at the starting line to 15 at the finish line in decimal values. .mem files contain the hexadecimal version of the same corrections.

Files are organized into two directories: FacingChambers and OffSideChambers. Facing chambers are parallel in phi, OffSideChambers are alternating in phi. Furthermore, files are differentiating between even and odd chamber numbers for the CSC chambers and between matches to the GEM layers closer to beamspot in |z| as layer1 and closer to the CSC as layer2 within the same chambers.

For the full correction, the sign of the slope and the z sign of the endcap need to be taken into account to determine the correction of the shift.
