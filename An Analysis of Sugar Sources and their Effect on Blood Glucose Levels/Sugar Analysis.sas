data Sugars;
 input A B $ C Y;
 label
 	   A='Subject'
	   B='Gender'
	   C='Sugar'
	   Y='Percent Change';
cards;
	1 M 1 28.13
	1 M 2 12.00
	1 M 3 5.68
	1 M 4 5.56
	1 M 5 66.25
	1 M 6 2.47
	2 M 1 5.71
	2 M 2 32.73
	2 M 3 10.20
	2 M 4 9.18
	2 M 5 13.41
	2 M 6 12.50
	1 F 1 12.75
	1 F 2 15.56
	1 F 3 14.12
	1 F 4 5.26
	1 F 5 21.11
	1 F 6 11.43
	2 F 1 26.09
	2 F 2 6.38
	2 F 3 0.00
	2 F 4 1.75
	2 F 5 27.50
	2 F 6 4.21
;

************************* Descriptive Statistics for response variable Y ****************************************;

proc means mean min max std data = Sugars;
	Title 'Descriptive Statistics for the Percent in Blood Sugar Change';
	var Y;
run;

proc sort data = Sugars out = boxplot_data;
	by A;
run;

proc boxplot data = boxplot_data;
	ods graphics off;
	Title 'Box Plot with Descriptive Statistics for Percent Blood Sugar Change';
	plot Y*A;
	 /*
	inset min mean max stddev /
		header = 'Overall Statistics for Response'
		pos    = tm;
	*/
	insetgroup min max mean/
		header = 'Extremes by Subject';
run;

proc sort data = Sugars out = boxplot_data;
	by B;
run;

proc boxplot data = boxplot_data;
	Title 'Box Plot with Descriptive Statistics for Percent Blood Sugar Change';
	plot Y*B;
	/*
	inset min mean max stddev /
		header = 'Overall Statistics for Response'
		pos    = tm;
	*/
	insetgroup min max mean/
		header = 'Extremes by Gender';
run;

proc sort data = Sugars out = boxplot_data;
	by C;
run;

proc boxplot data = boxplot_data;
	Title 'Box Plot with Descriptive Statistics for Percent Blood Sugar Change';
	plot Y*C;
	/*
	inset min mean max stddev /
		header = 'Overall Statistics for Response'
		pos    = tm;
	*/
	insetgroup min max mean/
		header = 'Extremes by Sugar Type';
run;

ods graphics on;

************************************* Levene Test using one way ANOVA and PROC GLM ********************************************;

proc glm data = Sugars;
	Title1 'Levene test for Gender';
	Title2 'A=Subject B=Gender C=Sugar Y=Percent Change';
	class B;
	model Y = B;
	means B/hovtest;
run;

proc glm data = Sugars;
	Title1'Levene test for Sugar Type';
	Title2 'A=Subject B=Gender C=Sugar Y=Percent Change';
	class C;
	model Y = C;
	means C/hovtest;
run;

************************************* Repeated Measure Analysis ********************************************;

proc glm data=Sugars;
	Title1 'Repeated Measure Analysis';
	Title2 'A=Subject B=Gender C=Sugar Y=Percent Change';
	class A B C;
	model Y = B A(B) C B*C;
	test h=B e=A(B);
	output out=sugar_out r=resid p=yhat;
run;

************************************* Normality Tests and Plot ********************************************;

proc univariate normal data = sugar_out;
    Title1 'Normal Tests for Sugar Experiment';
	Title2 'A=Subject B=Gender C=Sugar Y=Percent Change';
    var resid; 
run;

proc rank normal=vw data = sugar_out; 
	/* Computing ranked normal scores by residuals*/
	var resid;
	ranks nscore; 
run;

proc plot;
	Title1 'Normal Plot for Sugar Experiment';
	Title2 'A=Subject B=Gender C=Sugar Y=Percent Change';
	plot resid*nscore='R'; /*plotting ranked residual vs. normal score*/
	label nscore='Normal Score'; 
run;
quit;

************************************* Inference: Pairwise Multiple Comparison with Tukey********************************************;

Proc GLM;
	Title1 'Tukey Multiple Comparison for Factor B';
	Title2 'A=Subject B=Gender C=Sugar Y=Percent Change';
	class A B C;
	model Y = B A(B) C B*C;
	means B / tukey cldiff;
run;

Proc GLM;
	Title1 'Tukey Multiple Comparison for Factor C';
	Title2 'A=Subject B=Gender C=Sugar Y=Percent Change';
	class A B C;
	model Y = B A(B) C B*C;
	means C / tukey cldiff;
run;

Proc GLM;
	Title1 'Estimated Mean and Standard Deviation of B(A)';
	Title2 'A=Subject B=Gender C=Sugar Y=Percent Change';
	class A B C;
	model Y = B A(B) C B*C;
	means A(B) / tukey cldiff;
run;
