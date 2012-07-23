{
 float E;
 float x, y, z;
 float dx, dy, dz;
 float weight;
 char volumeName[64];
 char particleName[64];
 char procName[64];

 TChain *  T = new TChain("PhaseSpace");
 
// select one of the four following phase spaces you wish to visualize:

 //T->Add("output/BeamLineEntrance.root");
 //T->Add("output/BeamLineMiddle.root");
 //T->Add("output/BeamLineExit.root");
 T->Add("output-1/GlobalBoxEntrance.root");
  
 T->SetBranchAddress("ParticleName",&particleName);
 T->SetBranchAddress("ProductionVolume",&volumeName);
 T->SetBranchAddress("ProductionProcess",&procName);
 T->SetBranchAddress("Ekine",&E);
 T->SetBranchAddress("X",&x);
 T->SetBranchAddress("Y",&y);
 T->SetBranchAddress("Z",&z);
 T->SetBranchAddress("dX",&dx);
 T->SetBranchAddress("dY",&dy);
 T->SetBranchAddress("dZ",&dz);
 T->SetBranchAddress("Weight",&weight);
 
 TH2D *histoEmittanceXTheta;
 histoEmittanceXTheta = new TH2D("histoEmittanceXTheta", "Emittance X #theta",100, -30, 30, 100, -30, 30);

 TH2D *histoEmittanceYPhi;
 histoEmittanceYPhi = new TH2D("histoEmittanceYPhi", "Emittance Y #phi",100, -30, 30, 100, -30, 30);

 TH2D *histoSpotSize;
 histoSpotSize = new TH2D("histoSpotSize", "Spot",100, -20, 20, 100, -20, 20);

 TH1D *histoEnergy;
 histoEnergy = new TH1D("histoEnergy", "Energy",100, 175, 185);

 int n = T->GetEntries();
 cout<<"\nNb Particles dans le PhS = "<<n<<endl;

for (int i=0; i<n; i++)
{   T->GetEntry(i);
  histoEmittanceXTheta->Fill(x,atan(dx/dz)*1000.,weight); // x-axis in mm and y-axis in mrad
  histoEmittanceYPhi->Fill(y,atan(dy/dz)*1000,weight);
  histoSpotSize->Fill(x,y,weight);
  histoEnergy->Fill(E,weight);
}
	gStyle->SetOptFit(1111);
	gStyle->SetPalette(1);
TCanvas * fenetre1 = new TCanvas("fenetre1","fefe",1);
fenetre1->Divide(2,2);
fenetre1->cd(1);
histoEmittanceXTheta->Draw("COLZ");
fenetre1->cd(2);
histoEmittanceYPhi->Draw("COLZ");
fenetre1->cd(3);
histoSpotSize->Draw("COLZ");
fenetre1->cd(4);
histoEnergy->Draw();
}
