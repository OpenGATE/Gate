void DrawBranches() {


    gStyle->SetTitleFontSize(0.0);
    gStyle->SetFrameBorderMode(0);
    gStyle->SetCanvasColor(0);
    gStyle->SetStatBorderSize(0);
//    gStyle->SetOptStat("nemMrR");
    gStyle->SetOptStat("nemr");

    gROOT->ForceStyle();

  TCanvas *canvas = new TCanvas("canvas"); 

  canvas->Print("validation_reference.ps[");

  TFile *f = new TFile("./OpticalSimulation.root"); 
  TTree *t = (TTree*)f->Get("OpticalData"); 

  cout << t->GetEntries() << endl;

  // -- Get list of branches for the tree
  TObjArray *o = t->GetListOfBranches(); 
  int m = o->GetEntries(); 
  cout << "Number of branches: " << m << endl;
  
  // -- Loop over all, and draw their variables into TCanvas c1
  int cnt(0); 
  for (int i = 0; i < m; ++i) {
//    canvas->cd(i+1);
    cnt = t->Draw(((TBranch*)(*o)[i])->GetName());

    canvas->Update(); 
    canvas->Print("validation_reference.ps");
  }
//    canvas->cd(0);
canvas->Print("validation_reference.ps]");

}

