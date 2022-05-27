unset multiplot
reset
load 'moreland.gp'
eval setdef('tex','0')

output='phase-space-svd-sparse-w-1234'
if (tex == 1) {
  set lmargin at screen 0
  set rmargin at screen 1
  set bmargin at screen 0
  set tmargin at screen 1
  set term lua tikz standalone createstyle size 3in,3in  #fontscale 0.6
  set output output.'.tex'
}

fn='phase_space_svd_sparse_beta_14'
n1 = "3"
n_all="4 5 6 8 10"

xval(i) = i * dbeta + betafrom

###
# do stuff

set xlabel '$\beta$'
set ylabel '$\langle \Psi_{\beta} | P_{1,2,3,4} | \Psi_{\beta} \rangle$'
set xrange [-1:0.8]
set yrange [0.8:1]
#set yrange [-0.05:1]
#set ytics 0,0.1,1
#set grid xtics ytics mytics

set key center center spacing 2

sumsqr(a,b,c,d)=a**2 + b**2 + c**2 + d**2

p fn.'_'.n1.'.dat' u 1:2 w linesp pointi 5 tit sprintf('$n = %s$', n1), \
  for [n in n_all] fn.'_'.n.'.dat' u 1:(sumsqr($3,$4,$5,$6)) w linesp pointi 5 tit sprintf('$%s$', n)
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
