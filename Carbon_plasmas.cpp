#include <iostream>
using namespace std;

int main()

{
	int iterations;
	int counter;
	double contC=0.166*6.02E+23;
	double contC2=0.166*6.02E+23;
	double contC3=1.66*6.02E+23;
	double contHe=16.6*6.02E+23;
	double k1=5.346E-18;
	double k2=2.47E-17;
	double k3=3.72E-23;
	double k4=2.63E-16;
	double k5=3.38E-20;
	double k6=2.63E-16;
	double dcontC;
	double dcontC2;
	double dcontC3;
	cout<<"number of iterations"<<endl;
	cin>>iterations;
	cout<<contC<<"is the initial concentration of C"<<endl;
	cout<<contC2<<"is the initial concentration of C2"<<endl;
	cout<<contC3<<"is the initial concentration of C3"<<endl;
	
	for(counter=1; counter<=iterations;counter++)
	{
		dcontC=1E-9*(-k1*contC*contC3+k2*contC2*contC2+2*k3*contC2*contHe-2*k4*contC*contC*contHe+k5*contC3*contHe-k6*contC*contC2*contHe);
		dcontC2=1E-9*(2*k1*contC*contC3-2*k2*contC2*contC2-k3*contC2*contHe+k4*contC*contC*contHe+k5*contC3*contHe-k6*contC*contC2*contHe);
		dcontC3=1E-9*(-k1*contC*contC3+k2*contC2*contC2-k5*contC3*contHe+k6*contC*contC2*contHe);
		contC=contC-dcontC;
		contC2=contC2-dcontC2;
		contC3=contC3-dcontC3;
		cout<<"concentration of C is"<<contC<<endl;
		cout<<"rate of change in C is"<<dcontC<<endl;
		cout<<"concentration of C2 is"<<contC2<<endl;
		cout<<"rate of change in C2 is"<<dcontC2<<endl;
		cout<<"concentration of C3 is"<<contC3<<endl;
		cout<<"rate of change in C3 is"<<dcontC3<<endl;
	}

	char dummy;
	cout<<"hit enter to quit program"<<endl;
	cin.ignore();
	dummy=cin.get();
	return 0;
}
