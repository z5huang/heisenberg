unset multiplot
reset
load 'moreland.gp'
eval setdef('tex','0')

output='jd-gap'
if (tex == 1) {
  set lmargin at screen 0
  set rmargin at screen 1
  set bmargin at screen 0
  set tmargin at screen 1
  set term lua tikz standalone createstyle size 3in,3in  #fontscale 0.6
  set output output.'.tex'
}

dat10 = 'erg_10_noproj.dat'
dat12 = 'erg_12_noproj.dat'

###
# do stuff

set xlabel '$D/J$'
set xrange [-2:2]
set ylabel '$\Delta/NJ$'
set grid

set key spacing 2

p dat10 u 1:(($3-$2)/10) w linesp tit '$N=10$', \
  dat12 u 1:(($3-$2)/12) w linesp tit '$N=12$'

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
