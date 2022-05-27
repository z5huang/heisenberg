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
  set term lua tikz standalone createstyle size 4in,3in  #fontscale 0.6
  set output output.'.tex'
}

fn='phase_space_svd_sparse_beta_14'
dat = fn.'.dat'

###
# do stuff

set xlabel '$\beta$'
set ylabel '$\left| \langle \Psi_{\beta} | U_{k}\rangle \right|$'
set xrange [-1:0.8]
set yrange [-0.05:1]
set ytics 0,0.1,1
set grid xtics ytics mytics

set key at first 0.6, first 0.8 spacing 2

p dat u 1:(abs($3)) w linesp pointi 5 tit '$k=1$', \
  for [i=4:6] '' u 1:(abs(column(i))) w linesp pointi 5 tit sprintf('$%d$', i-2), \
  '' u 1:(sqrt($2)) w linesp lc 7 pointi 5 tit '$1,2,3,4$'

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
