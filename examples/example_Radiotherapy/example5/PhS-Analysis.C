{
 float E;
 float x, y, z;
 float dx, dy, dz;
 float weight;
 char volumeName[64];
 char particleName[64];
 char procName[64];

 TChain *  T = new TChain("PhaseSpace");
 
 T->Add("output/GlobalBoxEntrance2Beams.root");
  
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
 
 TH2D *histoSpotSize;
 histoSpotSize = new TH2D("histoSpotSize", "Spot",100, -200, 200, 100, -100, 100);

 int n = T->GetEntries();
 cout<<"\nNb Particles dans le PhS = "<<n<<endl;

for (int i=0; i<n; i++)
{   T->GetEntry(i);
  histoSpotSize->Fill(x,z,weight);
}
	gStyle->SetOptFit(1111);
	gStyle->SetPalette(1);
TCanvas * fenetre1 = new TCanvas("fenetre1","fefe",1);
fenetre1->cd(1);
histoSpotSize->Draw("COLZ");
}

