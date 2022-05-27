unset multiplot
reset
load 'moreland.gp'
eval setdef('tex','0')

output='proj_0pi'
if (tex == 1) {
  set lmargin at screen 0
  set rmargin at screen 1
  set bmargin at screen 0
  set tmargin at screen 1
  set term lua tikz standalone preamble '\usepackage{amsmath}' createstyle size 4in,2in  #fontscale 0.6
  set output output.'.tex'
}

fn='proj_0pi'
dat=fn.'.dat'
#par=fn.'.par'
#load par

###
# do stuff

set xlabel '$k/\pi$'
set xrange [0:2]
set xtics 0,1,2
set mxtics 2

set ylabel 'Angle ($\pi$)'
set yrange [-1:1]
set ytics -1,1,1 nomirror
set mytics 2

set y2label 'Total weight'
set y2range [0.94:1]
set y2tics 0.94,0.03,1
set my2tics 3

set key spacing 2 

p dat u ($1/pi):($2**2 + $3**2) axis x1y2 w linesp ps 2 tit '$\langle \Psi(k) | P | \Psi(k)\rangle$', \
  '' u ($1/pi):($6/pi) w linesp ps 2 tit '$\theta$',\
  '' u ($1/pi):($7/pi) w linesp ps 2 tit '$\varphi$', \


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
