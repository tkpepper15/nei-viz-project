function resnorm = checkResnorm(Circuit1, Circuit2)
% Function that takes two circuit elements and returns a resnorm over a
% large frequency range (fixed).

freq_range = logspace(-1, 5, 100);

[Zreal1, Zimag1] = impedanceCheck(...
    Circuit1(1), Circuit1(2), Circuit1(3), Circuit1(4), Circuit1(5), freq_range);

[Zreal2, Zimag2] = impedanceCheck(...
    Circuit2(1), Circuit2(2), Circuit2(3), Circuit2(4), Circuit2(5), freq_range);

resnorm = sum(sqrt((Zreal1-Zreal2).^2+(Zimag1-Zimag2).^2)) / (length(freq_range));