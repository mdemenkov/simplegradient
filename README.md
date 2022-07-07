Simple unconstrained gradient method with Armijo rule in Julia 

***Example of usage***

  Q=[2.25144   0.94941  -0.972442 <br>
   0.94941   2.51176   1.57232 <br>
  -0.972442  1.57232   2.2813]

*using* SmoothFunctions

F=Quadratic(Q) <br>
par=ArmijoParams(1,0.5,0.1)

x0=[1.;1.;1.] <br>
xmin=GradientMethod(F,x0,par)
