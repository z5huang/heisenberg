unset multiplot
reset
load 'moreland.gp'
eval setdef('tex','0')

output='proj-ent-12'
if (tex == 1) {
  set lmargin at screen 0
  set rmargin at screen 1
  set bmargin at screen 0
  set tmargin at screen 1
  set term lua tikz standalone createstyle size 4in,2in  #fontscale 0.6
  set output output.'.tex'
}

fn='proj_ent_12'
dat=fn.'.dat'
par=fn.'.par'
load par

###
# do stuff

es_lev_max = 1+3 + 5+3+3+5 + 1

# nex = -1 # exact state
nex = -1

set xrange [0.2:0.4]
#set xrange [-1:0.8]
#set xtics 0.2
#set xlabel '$\beta$'
#
#set yrange [-0.01:0.7]
#set ytics 0,0.05
#set ylabel 'Entanglement Spectrum'

p for [es = 1:es_lev_max] dat every es_cutoff::(es - 1) u 1:(log(column(4+nex))) w linesp pointi es tit sprintf('%d',es)

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
