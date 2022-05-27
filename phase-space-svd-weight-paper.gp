unset multiplot
reset
load 'moreland.gp'
eval setdef('tex','0')

output='phase-space-svd-weight'
if (tex == 1) {
  set lmargin at screen 0
  set rmargin at screen 1
  set bmargin at screen 0
  set tmargin at screen 1
  set term lua tikz standalone createstyle size 2in,2in  #fontscale 0.6
  set output output.'.tex'
}

fn='phase_space_svd'
dat14=fn.'_14_d.dat'
dat12=fn.'_d.dat'
M=180
#load par

###
# do stuff

set xlabel '$\kappa$'
#set ylabel '$w_\kappa$'
set xrange [1:8]
set xtics 1,1,8
set yrange [0:0.8]
set ytics 0,0.1,1
set grid xtics ytics mytics
set key spacing 2

set key at first 7, first 0.6
p dat14 u ($0 + 1):(($1)**2 / M) w linesp pt 2 tit '\huge{$w_\kappa$}' #notit # '$N=14$'
#  dat12 u ($0 + 1):(($1)**2 / M) w linesp pt 6 tit '$N=12$

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
