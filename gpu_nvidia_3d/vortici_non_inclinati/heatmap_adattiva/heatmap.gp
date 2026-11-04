set terminal pngcairo size 1100,850 enhanced font ",11"
set output "/home/zorka/simulazioni/Test 1/gpu_nvidia_3d/vortici_non_inclinati/heatmap_adattiva/heatmap.png"
set title "log10 (|vort| error) - t = 30" font ",13"
set xlabel "1 / VORTERR"
set ylabel "MAXLEVEL"
set xtics ( "3e+00" 1, "1e+01" 2, "3e+01" 3,)
set ytics 1
set xrange [0.5:3.5]
set yrange [3.5:8.5]
set palette defined (0 "#440154", 0.25 "#3b528b", 0.5 "#21918c", 0.75 "#5ec962", 1 "#fde725")
set cbrange [-5.99:-0.35]
set colorbox
set cblabel "log10 (||vort - vort_ref||_inf)" font ",11"
unset key
plot "/home/zorka/simulazioni/Test 1/gpu_nvidia_3d/vortici_non_inclinati/heatmap_adattiva/heatmap.dat" using 1:2:3 with image, \
     "/home/zorka/simulazioni/Test 1/gpu_nvidia_3d/vortici_non_inclinati/heatmap_adattiva/heatmap.dat" using 1:2:(($5==1) ? \
              sprintf("%s\n(%d)", "ref", $4) : \
              sprintf("%.2f\n(%d)", $3, $4)) \
          with labels font ",9"
