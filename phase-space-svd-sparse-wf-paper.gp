unset multiplot
reset
load 'moreland.gp'
eval setdef('tex','0')

output='phase-space-svd-sparse-wf'
if (tex == 1) {
  set lmargin at screen 0
  set rmargin at screen 1
  set bmargin at screen 0
  set tmargin at screen 1
  set term lua tikz standalone createstyle size 2in,2in  #fontscale 0.6
  set output output.'.tex'
}

dat='phase_space_svd_sparse_beta_14.dat'
dat_w = 'phase_space_svd_sparse_beta_14_sval.dat'
#dat=fn.'_ud.dat'
#par=fn.'.par'

nlev = 4
#load par

#xval(i) = i * dbeta + betafrom

###
# do stuff

set multiplot
set origin 0,0
set size 1,1



set xlabel '$\lambda$'
#set ylabel '$\left| \langle \Psi_{\lambda} | U_{k}\rangle  \right|$'
#set xrange [betafrom:betato]
set xrange [-1:0.8]
#set yrange [-0.05:1]
set yrange [0:1]
set ytics 0,0.1,1
set grid xtics ytics mytics
set xtics -1,0.5
set xtics add (0.8)
set mxtics 5
set ytics 0,0.5
set mytics 5

#set key at first 0.6, first 0.86 spacing 1.5 reverse Left  #samplen 1  #maxcol 2 maxrow 3
set key at first 0.73, first 0.86 spacing 1.5 reverse Left  samplen 1  #maxcol 2 maxrow 3

#p for [i=1:4] dat  u 1:(abs(column(i+2))) w linesp pointi 6 lt i tit sprintf('$f_{%d}$', i), \
#  dat u 1:(sqrt(column(3)**2 + column(4)**2 + column(5)**2 + column(6)**2)) w linesp ps 0.5 pt 5 pointi 6 lc rgb color='red' tit '\tiny{$\sqrt{f_1^2 + f_2^2 + f_3^2 + f_4^2}$}', \
#  0.99 w l lt 1 dashtype 2 lc -1 notit
p for [i=1:4] dat  u 1:(abs(column(i+2))) w linesp pointi 6 lt i tit sprintf('$f_{%d}$', i), \
  dat u 1:(sqrt($2)) w linesp ps 0.5 pt 5 pointi 6 lc rgb color='red' tit '\tiny{$\sqrt{f_1^2 + f_2^2 + f_3^2 + f_4^2}$}', \
  0.99 w l lt 1 dashtype 2 lc -1 notit


unset lmargin
unset rmargin
unset tmargin
unset bmargin
set origin 0.15,0.48
set size 0.6,0.48
set xrange [1:8]
#set xtics 1,8 format '' #format sprintf('\tiny{%d}')
#set mxtics 8
set xlabel sprintf('\small{$\kappa$}') offset 0, 1
set xtics 1,1,8 format '' font ', 5'
unset mxtics
set x2tics 1,1,8 font ', 5' offset 0, -0.5

#set yrang [0:0.8]
#set ytics 0,0.4,0.8 format ''
set yrange [0:1]
set ytics 0,0.5, 1 font ', 5' offset first 1, first 0
set mytics 2
#set mytics 4
#set y2tics 0,0.4,0.8 font ', 5'

unset grid
set key at 7.5,0.9
p dat_w u ($0+1):($2**2/4) w linesp pt 6 ps 0.5 tit sprintf('\small{$w_\kappa$}')

unset multiplot

#
###

if (tex == 1){
  unset output
  set term wxt
  build = buildtex(output)
  print '"eval build" to build and preview'
} else {
  #print "press enter to continue"
  #pause -1
}
