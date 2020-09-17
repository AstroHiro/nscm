clear all
close all
a = 10;
b = 1;
c = b;
d = 1;
gc = 2;
fun = @(e) (a+b./e)./(gc-(c.*e+d))
es = (-b+sqrt(b^2+a*(gc-d)))/a;
e = 0.001:0.001:(gc-d)/c-0.01;
figure
plot(e,fun(e))
hold on
scatter(es,fun(es))

L = 6;
g = 10
fun = @(x) x.^(L+2).*(x.^(L)-1)./(x-1)-g
figure
fplot(fun,[0.1,1.3])
x = fsolve(fun,2)