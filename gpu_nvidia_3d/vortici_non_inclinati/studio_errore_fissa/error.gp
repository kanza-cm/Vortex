set terminal pngcairo size 1000,700 enhanced font ",12"
set output "error.png"
set grid
set xlabel "MAXLEVEL"
set ylabel "log10 ||wz_L - wz_ref||_{inf}   (fissa 3D GPU, CFL 0.5, ref = L8)"
set key top left
plot "error.dat" u 1:(log10($2)) w lp lw 2 pt 7 t "||.||_{inf}", \
     "error.dat" u 1:(log10($3)) w lp lw 2 pt 5 t "||.||_2"
