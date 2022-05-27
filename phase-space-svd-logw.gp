unset multiplot
reset
load 'moreland.gp'
eval setdef('tex','0')

output='phase-space-svd-logw'
if (tex == 1) {
  set lmargin at screen 0
  set rmargin at screen 1
  set bmargin at screen 0
  set tmargin at screen 1
  set term lua tikz standalone createstyle size 6in,3in  #fontscale 0.6
  set output output.'.tex'
}

fn='phase_space_svd'
dat14=fn.'_14_d.dat'
dat12 = fn.'_d.dat'
M=180
#load par

###
# do stuff

set xlabel '$k$'
set ylabel '$\log(w_k)$'
set xrange [1:]
set yrange [-0.4:0.05]
set ytics -0.4,0.1,0
set mytics 2
set ytics add (0.05)
set xtics 0,20
set mxtics 4
set xtics add (1)
set grid xtics mxtics ytics
set key spacing 2
#set yrange [-0.05:1]
#set ytics 0,0.1,1
#set grid xtics ytics mytics

#set logscale y
p dat14 u ($0 + 1):(log(($1)**2) / M) w linesp pt 2 tit '$N=14$', \
  dat12 u ($0 + 1):(log(($1)**2) / M) w linesp pt 6 tit '$N=12$'
#p dat u ($0 + 1):((($1)**2) / M) w linesp pt 2 notit

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
