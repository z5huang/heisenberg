unset multiplot
reset
load 'moreland.gp'
eval setdef('tex','0')

output='var-proj-fidelity'
if (tex == 1) {
  set lmargin at screen 0
  set rmargin at screen 1
  set bmargin at screen 0
  set tmargin at screen 1
  set term lua tikz standalone createstyle size 3in,3in  #fontscale 0.6
  set output output.'.tex'
}

fn='var_proj_consecutive_14'
#fn='var_proj_consecutive'
dat=fn.'_fidelity.dat'
par=fn.'.par'
load par

###
# do stuff
set xlabel '$\beta$'
set ylabel 'Fidelity'
set key at first 0.6,first 0.6 spacing 2
set grid
set yrange [0.2:1.02]

#set logscale y

p dat u 1:2 w linesp pointint 5 tit 'AKLT', \
  for [i=1:nex_max] '' u 1:( (column(2+i)) ) w linesp pointi 5 tit sprintf('$v=%d$', i)
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
