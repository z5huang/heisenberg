unset multiplot
reset
load 'moreland.gp'
eval setdef('tex','0')

output='h-inter-gap'
if (tex == 1) {
  set lmargin at screen 0
  set rmargin at screen 1
  set bmargin at screen 0
  set tmargin at screen 1
  set term lua tikz standalone createstyle size 3in,3in  #fontscale 0.6
  set output output.'.tex'
}

fn='h_inter_gap'
dat=fn.'.dat'
par=fn.'.par'
load par

###
# do stuff

set xrange [beta_from-dbeta:beta_to+dbeta]
set xlabel '$\beta$'
set yrange [0 - 2.0/nk:2]
set ylabel '$k/\pi$'
set grid front

set size ratio -1

set xyplane 0.1
set hidden3d
#sp dat u 1:($2/pi):(abs($3-$4)) w linesp notit , 0 notit
p dat u 1:($2/pi):(abs($3-$4)) w image pixel notit


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
