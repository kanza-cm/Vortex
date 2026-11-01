#!/bin/sh
# Full study of the 2nd-order solver (AB2, CFL=0.8 by default) for
# the merging of two vortices. It unifies the old studies, rewritten
# at 2nd order:
#
#   fissa    - fixed grid error   (L=LMIN..LMAX, reference LMAX)
#   adattiva - adaptive grid error (LMIN..LMAX, reference LMAX)
#   heatmap  - map of log10(err) over (MAXLEVEL x EPS)
#   movie    - movie of the adaptive grid for each L
#   livelli  - refinement level + vorticity panels for each L
#   plot     - palinstrophy and nuclei separation (from the fissa studies)
#
# Usage:  ./studio_completo_2nd.sh [phases...]     (default: all)
#   e.g.: ./studio_completo_2nd.sh fissa plot
#   e.g.: NP=16 LMIN=6 LMAX=10 ./studio_completo_2nd.sh
#
# Output:
#   studio_completo/heatmap/               heatmap.dat / heatmap.png
#   studio_completo/movie/L<L>/            griglia.mp4
#   studio_completo/livelli/L<L>/          level.png / omega.png
#   studio_completo/palinstrophy_separation.png
#   error_study_fissa_2nd/ and error_study_adattiva_2nd/  (error studies)
set -e
cd "$(dirname "$0")"

export BASILISK="${BASILISK:-$HOME/basilisk/src}"
export PATH="$PATH:$BASILISK"

if [ $# -gt 0 ]; then PHASES="$*"; else PHASES="fissa adattiva heatmap movie livelli plot"; fi

has() {
  case " $PHASES " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

LMIN="${LMIN:-6}"            # first level / MINLEVEL
LMAX="${LMAX:-10}"           # last level = reference (fissa/adattiva)
T="${T:-200}"                # duration of the error studies
NP="${NP:-16}"                # MPI processes
CFL="${CFL:-0.8}"            # solver CFL
NIT="${NIT:-6}"              # fixed Poisson iterations
EPS0="${EPS0:-1e-3}"         # reference wavelet tolerance

HEAT_T="${HEAT_T:-20}"       # heatmap run duration
EPS_LIST="${EPS_LIST:-1e-3 5e-4 2.5e-4 1.25e-4}"
TMAX_MOV="${TMAX_MOV:-100}"  # movie/livelli duration
SNAP="${SNAP:-5}"            # grid frame every SNAP
NSNAP="${NSNAP:-6}"          # level panels (6 = 2x3)
GRID_SNAP="${GRID_SNAP:-1024}"
FPS="${FPS:-10}"
OMEGA0="${OMEGA0:-1}"

out="studio_completo"
mkdir -p "$out"

echo "== studio_completo_2nd   fasi: $PHASES =="

# ---------------- 1) fixed grid error ----------------
if has fissa; then
  echo
  echo "-- fissa: error study a griglia fissa (riferimento L$LMAX) --"
  LMIN="$LMIN" LMAX="$LMAX" T="$T" NP="$NP" NIT="$NIT" CFL="$CFL" \
      ./error_study_fissa_2nd.sh
fi

# ---------------- 2) adaptive grid error ----------------
if has adattiva; then
  echo
  echo "-- adattiva: error study a griglia adattiva (riferimento L$LMAX) --"
  LMIN="$LMIN" LMAX="$LMAX" T="$T" NP="$NP" NIT="$NIT" EPS0="$EPS0" CFL="$CFL" \
      ./error_study_adattiva_2nd.sh
fi

# ---------------- 3) heatmap error ----------------
if has heatmap; then
  echo
  echo "-- heatmap: errore su (MAXLEVEL x EPS), T=$HEAT_T --"
  hm="$out/heatmap"
  rm -rf "$hm"; mkdir -p "$hm"
  for EPS in $EPS_LIST; do
    L=$LMIN
    while [ "$L" -le "$LMAX" ]; do
      run="$hm/L${L}_eps${EPS}"
      mkdir -p "$run"
      if [ "$L" -le 8 ]; then NPL=4; else NPL=$NP; fi
      extra=""
      [ -n "$NIT" ] && extra="-DPOISSON_ITER=$NIT"
      echo "   === L=$L  EPS=$EPS  (np=$NPL) ==="
      qcc -source -grid=quadtree -D_MPI=$NPL -O3 -march=native \
          -DCFL_USER=$CFL -DMOVIE=0 -DMINLEVEL=$LMIN -DMAXLEVEL=$L \
          -DEPS=$EPS -DTMAX=$HEAT_T -DMATRIX_N=$((1 << L)) $extra \
          saetta_adattiva.c
      mpicc -O3 -march=native -Wall -Wno-array-bounds -Wno-stringop-overflow \
          -o "$run/saetta" _saetta_adattiva.c -lm
      rm -f _saetta_adattiva.c
      ( cd "$run" && mpirun --oversubscribe -np $NPL ./saetta > log.txt 2>&1 )
      L=$((L + 1))
    done
  done

  ref_eps=$(echo "$EPS_LIST" | awk '{print $NF}')
  ref="$hm/L${LMAX}_eps${ref_eps}/omega_matrix.bin"
  [ -f "$ref" ] || { echo "manca il riferimento $ref"; exit 1; }

  echo "# x_idx L log10_err celle ref" > "$hm/heatmap.dat"
  L=$LMIN
  while [ "$L" -le "$LMAX" ]; do
    jx=1
    for EPS in $EPS_LIST; do
      dir="$hm/L${L}_eps${EPS}"
      cells=$(awk '$1 ~ /^[0-9]/ {print $18}' "$dir/diagnostics.dat" | tail -1)
      if [ "$L" = "$LMAX" ] && [ "$EPS" = "$ref_eps" ]; then
        echo "$jx $L NaN $cells 1" >> "$hm/heatmap.dat"
        echo "   L=$L EPS=$EPS -> RIFERIMENTO (celle=$cells)"
      else
        e=$(python3 ../error_norm.py "$dir/omega_matrix.bin" "$ref" | awk '{print $1}')
        le=$(awk "BEGIN{printf \"%.2f\", log($e)/log(10)}")
        echo "$jx $L $le $cells 0" >> "$hm/heatmap.dat"
        echo "   L=$L EPS=$EPS -> log10(err)=$le  celle=$cells"
      fi
      jx=$((jx + 1))
    done
    L=$((L + 1))
  done

  tics=""
  j=1
  for EPS in $EPS_LIST; do
    lab=$(awk "BEGIN{printf \"%.0e\", 1/$EPS}")
    tics="$tics \"$lab\" $j,"
    j=$((j + 1))
  done
  nx=$((j - 1))
  rmin=$(awk '$5==0 && (mn=="" || $3<mn){mn=$3} END{print mn}' "$hm/heatmap.dat")
  rmax=$(awk '$5==0 && (mx=="" || $3>mx){mx=$3} END{print mx}' "$hm/heatmap.dat")

  cat > "$hm/heatmap.gp" <<EOF
set terminal pngcairo size 1100,850 enhanced font ",11"
set output "heatmap.png"
set title "log10 (L_{inf} error) - t = $HEAT_T" font ",13"
set xlabel "1/EPS"
set ylabel "MAXLEVEL"
set xtics ($tics)
set ytics 1
set xrange [0.5:$(awk "BEGIN{print $nx+0.5}")]
set yrange [$(awk "BEGIN{print $LMIN-0.5}"):$(awk "BEGIN{print $LMAX+0.5}")]
set palette defined (0 "#440154", 0.25 "#3b528b", 0.5 "#21918c", 0.75 "#5ec962", 1 "#fde725")
set cbrange [$(awk "BEGIN{print $rmin-0.1}"):$(awk "BEGIN{print $rmax+0.1}")]
set colorbox
set cblabel "log10 (||omega - omega_{ref}||_{inf})" font ",11"
unset key
plot "heatmap.dat" using 1:2:3 with image, \\
     "heatmap.dat" using 1:2:((\$5==1) ? \\
              sprintf("Reference solution\\n(%d)", \$4) : \\
              sprintf("%.2f\\n(%d)", \$3, \$4)) \\
          with labels font ",9"
EOF
  ( cd "$hm" && gnuplot heatmap.gp )
  echo "   heatmap: $hm/heatmap.png"
fi

# ---------------- 4) movie of the adaptive grid ----------------
if has movie; then
  echo
  echo "-- movie: griglia adattiva per L=$LMIN..$LMAX (TMAX=$TMAX_MOV) --"
  L=$LMIN
  while [ "$L" -le "$LMAX" ]; do
    mdir="$out/movie/L$L"
    rm -rf "$mdir"; mkdir -p "$mdir/cells" "$mdir/frames"
    eps=$(awk "BEGIN{printf \"%.3e\", $EPS0/2^($L-$LMIN)}")
    extra=""
    [ -n "$NIT" ] && extra="-DPOISSON_ITER=$NIT"
    echo "   === MAXLEVEL=$L  EPS=$eps  (np=$NP) ==="
    qcc -source -grid=quadtree -D_MPI=$NP -O3 -march=native \
        -DCFL_USER=$CFL -DMOVIE=0 -DMINLEVEL=$LMIN -DMAXLEVEL=$L \
        -DEPS0=$EPS0 -DTMAX=$TMAX_MOV -DSNAP=$SNAP $extra saetta_griglia.c
    mpicc -O3 -march=native -Wall -Wno-array-bounds -Wno-stringop-overflow \
        -o "$mdir/saetta" _saetta_griglia.c -lm
    rm -f _saetta_griglia.c
    ( cd "$mdir" && mpirun --oversubscribe -np $NP ./saetta > log.txt 2>&1 )

    n=$(awk "BEGIN{printf \"%d\", $TMAX_MOV/$SNAP}")
    i=0
    while [ "$i" -le "$n" ]; do
      tag=$(printf "cells_%05d" "$i")
      if ls "$mdir/cells/${tag}"_*.dat >/dev/null 2>&1; then
        cat "$mdir/cells/${tag}"_*.dat > "$mdir/frames/${tag}.dat"
      fi
      i=$((i + 1))
    done

    cat > "$mdir/frames.gp" <<EOF
set terminal pngcairo size 800,800 enhanced font ",10"
unset border; unset xtics; unset ytics; unset key
set xrange [-0.5:0.5]; set yrange [-0.5:0.5]
set size square
do for [i=0:$n] {
  set output sprintf("frames/frame_%04d.png", i)
  set title sprintf("griglia adattiva  -  t = %.0f", i*$SNAP)
  plot sprintf("frames/cells_%05d.dat", i) w l lc rgb "#333333" lw 0.3
}
EOF
    ( cd "$mdir" && gnuplot frames.gp )
    ( cd "$mdir" && ffmpeg -y -loglevel error -framerate "$FPS" \
        -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p griglia.mp4 )
    echo "   movie: $mdir/griglia.mp4"
    L=$((L + 1))
  done
fi

# ---------------- 5) level panels ----------------
if has livelli; then
  echo
  echo "-- livelli: pannelli livello+vorticità per L=$LMIN..$LMAX --"
  L=$LMIN
  while [ "$L" -le "$LMAX" ]; do
    ldir="$out/livelli/L$L"
    rm -rf "$ldir"; mkdir -p "$ldir/livelli"
    eps=$(awk "BEGIN{printf \"%.3e\", $EPS0/2^($L-$LMIN)}")
    step=$(awk "BEGIN{printf \"%.10g\", $TMAX_MOV/($NSNAP-1)}")
    extra=""
    [ -n "$NIT" ] && extra="-DPOISSON_ITER=$NIT"
    echo "   === MAXLEVEL=$L  EPS=$eps  (np=$NP) ==="
    qcc -source -grid=quadtree -D_MPI=$NP -O3 -march=native \
        -DCFL_USER=$CFL -DMOVIE=0 -DMINLEVEL=$LMIN -DMAXLEVEL=$L \
        -DEPS0=$EPS0 -DOMEGA0=$OMEGA0 -DTMAX=$TMAX_MOV -DNSNAP=$NSNAP \
        -DGRID_SNAP=$GRID_SNAP -DSNAP_STEP=$step $extra saetta_livelli.c
    mpicc -O3 -march=native -Wall -Wno-array-bounds -Wno-stringop-overflow \
        -o "$ldir/saetta" _saetta_livelli.c -lm
    rm -f _saetta_livelli.c
    ( cd "$ldir" && mpirun --oversubscribe -np $NP ./saetta > log.txt 2>&1 )

    rows=$(awk "BEGIN{print int(($NSNAP+2)/3)}")
    splist=""
    i=0
    while [ "$i" -lt "$NSNAP" ]; do
      splist="$splist
splot 'livelli/level-$i'"
      i=$((i + 1))
    done
    omlist=""
    i=0
    while [ "$i" -lt "$NSNAP" ]; do
      omlist="$omlist
splot 'livelli/omega-$i'"
      i=$((i + 1))
    done
    if [ "$LMIN" = "$LMAX" ]; then
      cmin=$(awk "BEGIN{print $LMIN-0.5}")
      cmax=$(awk "BEGIN{print $LMAX+0.5}")
    else
      cmin=$LMIN
      cmax=$LMAX
    fi

    cat > "$ldir/livelli.gp" <<EOF
set terminal pngcairo size 1100,800 enhanced font ",11"
set size ratio -1
unset key
unset xtics
unset ytics
unset border
set pm3d
set pm3d map interpolate 1,1
set palette rgbformulae 33,13,10

set output 'level.png'
set cbrange [$cmin:$cmax]
set multiplot layout $rows,3 scale 1.6,1.6
$splist
unset multiplot

set output 'omega.png'
set cbrange [0:$OMEGA0]
set multiplot layout $rows,3 scale 1.6,1.6
$omlist
unset multiplot
EOF
    ( cd "$ldir" && gnuplot livelli.gp )
    echo "   livelli: $ldir/level.png  $ldir/omega.png"
    L=$((L + 1))
  done
fi

# ---------------- 6) palinstrophy / separation plot ----------------
if has plot; then
  echo
  echo "-- plot: palinstrofia e separazione (dagli studi fissa) --"
  fdir="error_study_fissa_2nd"
  if [ ! -f "$fdir/run$LMIN/diagnostics.dat" ]; then
    echo "   ATTENZIONE: mancano le run in $fdir (esegui prima la fase 'fissa')"
  else
    cat > "$out/palinstrophy_separation.gp" <<EOF
set terminal pngcairo size 1100,1000 enhanced font ",14"
set grid
set key top left
set xlabel "t"
set xrange [0:$T]

set output "palinstrophy_separation.png"
set multiplot layout 2,1
set title "Palinstrofia  P = int |grad omega|^2 dA"
set ylabel "P"
plot for [L=$LMIN:$LMAX] sprintf("../$fdir/run%d/diagnostics.dat", L) \\
     using 1:5 with lines title sprintf("L%d", L)
set title "Separazione dei due nuclei"
set ylabel "d"
plot for [L=$LMIN:$LMAX] sprintf("../$fdir/run%d/diagnostics.dat", L) \\
     using 1:12 with lines title sprintf("L%d", L)
unset multiplot
EOF
    ( cd "$out" && gnuplot palinstrophy_separation.gp )
    echo "   plot: $out/palinstrophy_separation.png"
  fi
fi

echo
echo "Fatto. Output in $out/ (heatmap/, movie/, livelli/) e negli studi errore."
