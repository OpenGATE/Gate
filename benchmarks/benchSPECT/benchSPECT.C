{


TFile *f =  (TFile*)gROOT->GetListOfFiles()->FindObject("benchSPECT.root");
if (!f) {
	   f = new TFile("benchSPECT.root");
}

TH1 *total_nb_primaries = (TH1*)gDirectory->Get("total_nb_primaries");
TTree *Singles = (TTree*)gDirectory->Get("Singles");


Int_t           comptonPhantom;
Int_t           comptonCrystal;
Float_t         energy;
Char_t          comptVolName[40];
Float_t          scatter_phantom,scatter_null,scatter_movsource,scatter_compartment,scat ter_table;
Float_t          scatter_collim,scatter_shielding,primary_event,scatter_crystal;
Float_t          order1,order2,order3,order3,order4,ordersup,ordertot;

Singles->SetBranchAddress("comptonPhantom",&comptonPhantom);
Singles->SetBranchAddress("comptonCrystal",&comptonCrystal);
Singles->SetBranchAddress("energy",&energy);
Singles->SetBranchAddress("comptVolName",comptVolName);

TH1F *primary = new TH1F("primary","",100,0,0.2);
TH1F *sc_ph = new TH1F("sc_ph","",100,0,0.2);
TH1F *sc_cry = new TH1F("sc_mo","",100,0,0.2);
TH1F *sc_com = new TH1F("sc_com","",100,0,0.2);
TH1F *sc_ta = new TH1F("sc_ta","",100,0,0.2);
TH1F *sc_col = new TH1F("sc_col","",100,0,0.2);
TH1F *ener = new TH1F("ener","",100,0,0.2);

TH1F *o1 = new TH1F("o1","",100,0,0.2);
TH1F *o2 = new TH1F("o2","",100,0,0.2);
TH1F *o3 = new TH1F("o3","",100,0,0.2);
TH1F *o4 = new TH1F("o4","",100,0,0.2);
TH1F *osup = new TH1F("osup","",100,0,0.2);


Int_t nentries = Singles->GetEntries();
Int_t nbytes = 0;


for (Int_t i=0; i<nentries;i++) {
  nbytes += Singles->GetEntry(i);

  ener.Fill(energy);
  if (comptonPhantom == 0 && comptonCrystal == 0) {
    primary.Fill(energy);
    primary_event++;
  }
  if (comptonCrystal != 0 && comptonPhantom == 0) {
    sc_cry.Fill(energy);
    scatter_crystal++;
  }

  if (strcmp(comptVolName,"Phantom_phys")==  NULL||strcmp(comptVolName,"movsource_phys") == NULL) {
    scatter_phantom++;
    sc_ph->Fill(energy);
  }
  if (strcmp(comptVolName,"compartment_phys") == NULL) {
    scatter_compartment++;
    sc_com->Fill(energy);
  }
  if (strcmp(comptVolName,"table_phys") == NULL) {
    scatter_table++;
    sc_ta->Fill(energy);
  }
  if (strcmp(comptVolName,"collimator_phys") == NULL) {
    scatter_collim++;
    sc_col->Fill(energy);
  }

  if(comptonPhantom + comptonCrystal == 1) {
    o1.Fill(energy);
    order1++;
  }
  if(comptonPhantom + comptonCrystal == 2) {
    o2.Fill(energy);
    order2++;
  }
  if(comptonPhantom + comptonCrystal == 3) {
    o3.Fill(energy);
    order3++;
  }
  if(comptonPhantom + comptonCrystal == 4) {
    o4.Fill(energy);
    order4++;
  }
  if(comptonPhantom + comptonCrystal > 4) {
    osup.Fill(energy);
    ordersup++;
  }

  if(comptonPhantom != 0 || comptonCrystal != 0) {
    ordertot++;
  }

}


// **************************************** Plots  **********************************************

gStyle->SetPalette(1);
gStyle->SetOptStat(0);

// First Canvas

TCanvas cont("contours","contours",100,100,800,600);
cont.Divide(2,2);

cont.SetFillColor(0);
cont.cd(1);

ener->SetFillColor(2);
ener->SetFillStyle(3023);
ener->GetXaxis()->SetTitle("MeV");
ener.Draw();
TLatex *   tex = new TLatex(0.0169022,2486.14,"Total spectrum of  the detected events");
tex->SetLineWidth(2);
tex->Draw();

cont.SetFillColor(0);
cont.cd(2);

primary.Draw();
primary->SetLineColor(6);
primary->GetXaxis()->SetTitle("MeV");

sc_ph.Draw("same");
sc_ph ->SetLineColor(1);

sc_ta.Draw("same");
sc_ta ->SetLineColor(2);

sc_col.Draw("same");
sc_col ->SetLineColor(3);

sc_com.Draw("same");
sc_com ->SetLineColor(4);

sc_cry.Draw("same");
sc_cry ->SetLineColor(5);

TLegend *leg1 = new TLegend(0.2,0.6,0.6,0.85);
leg1->SetFillColor(0);
leg1->SetTextSize(0.03);
leg1->AddEntry(primary,"primary spectrum","l");
leg1->AddEntry(sc_ph,"scatter in the phantom","l");
leg1->AddEntry(sc_ta,"scatter in the table","l");
leg1->AddEntry(sc_col,"scatter in the collimator","l");
leg1->AddEntry(sc_com,"scatter in the backcompartment","l");
leg1->AddEntry(sc_cry,"scatter in the crystal","l");
leg1->Draw();

tex = new TLatex(-0.0040358,1741.43,"Primary and scatter spectra of  the detected events");
tex->SetLineWidth(2);
tex->Draw();

cont.SetFillColor(0);
cont.cd(3);

o1.Draw();
o1->GetXaxis()->SetTitle("MeV");
o1 ->SetLineColor(1);
o2.Draw("same");
o2 ->SetLineColor(2);
o3.Draw("same");
o3 ->SetLineColor(3);
o4.Draw("same");
o4 ->SetLineColor(4);
osup.Draw("same");
osup ->SetLineColor(5);

TLegend *leg2 = new TLegend(0.2,0.6,0.5,0.85);
leg2->SetFillColor(0);
leg2->SetTextSize(0.03);
leg2->AddEntry(o1,"1st order scatter","l");
leg2->AddEntry(o2,"2st order scatter","l");
leg2->AddEntry(o3,"3st order scatter","l");
leg2->AddEntry(o4,"4st order scatter","l");
leg2->AddEntry(osup,">4st order scatter","l");
leg2->Draw();

tex = new TLatex(0.022791,792.626,"Scatter spectra of the detected  events");
tex->SetLineWidth(2);
tex->Draw();

cont.SetFillColor(0);
cont.cd(4);

tex = new TLatex(0.0498325,0.368881,"benchmarkSPECT : AnaROOT  Analysis");
tex->SetTextColor(2);
tex->SetTextFont(72);
tex->SetTextSize(0.0728438);
tex->SetLineWidth(2);
tex->Draw();
tex = new TLatex(0.0812395,0.663899,"OpenGATE Collaboration");
tex->SetTextColor(4);
tex->SetTextSize(0.105624);
tex->SetLineWidth(2);
tex->Draw();

//****************************** TEXT *************************

// Number of emitted particles
cout<<"##### Number of emitted particles :  "<<total_nb_primaries->GetMean()<<endl;
// Number of detected counts from 20 to 190 keV
cout<<"##### Number of detected counts from 20 to 190 keV :  "<<nentries<<" events"<<endl;
// Primary events
cout<<"##### Primary events : "<<primary_event/nentries*100<<"  %"<<endl;
// Scatter in the phantom
cout<<"##### Scatter in the phantom :  "<<scatter_phantom/nentries*100<<" %"<<endl;
// Scatter in the table
cout<<"##### Scatter in the table :  "<<scatter_table/nentries*100<<" %"<<endl;
// Scatter in the collimator
cout<<"##### Scatter in the collimator :  "<<scatter_collim/nentries*100<<" %"<<endl;
// Scatter in the crystal
cout<<"##### Scatter in the crystal :  "<<scatter_crystal/nentries*100<<" %"<<endl;
// Scatter in the backcompartment
cout<<"##### Scatter in the backcompartment :  "<<scatter_compartment/nentries*100<<" %"<<endl;
// Scatter order : 1
cout<<"##### Scatter order 1 : "<<order1/ordertot*100<<" %"<<endl;
// Scatter order : 2
cout<<"##### Scatter order 2 : "<<order2/ordertot*100<<" %"<<endl;
// Scatter order : 3
cout<<"##### Scatter order 3 : "<<order3/ordertot*100<<" %"<<endl;
// Scatter order : 4
cout<<"##### Scatter order 4 : "<<order4/ordertot*100<<" %"<<endl;
// Scatter order : >4
cout<<"##### Scatter order >4 : "<<ordersup/ordertot*100<<"  %"<<endl;

FILE *fp = fopen("benchSPECTRA.txt","w");
fprintf(fp,"####################################################\n");
fprintf(fp,"#####                                           ####\n");
fprintf(fp,"#####         SPECT BENCHMARK RESULTS           ####\n");
fprintf(fp,"#####         OpenGATE Collaboration            ####\n");
fprintf(fp,"#####                                           ####\n");
fprintf(fp,"####################################################\n");
fprintf(fp,"\n");

fprintf(fp,"##### Number of emitted particles :  %8f\n",total_nb_primaries->GetMean());
fprintf(fp,"##### Number of detected counts from 20 to 190 keV :  %8f\n",nentries);
fprintf(fp,"##### Primary events : %8f\n",primary_event);
fprintf(fp,"##### Scatter in the phantom : %8f\n",scatter_phantom);
fprintf(fp,"##### Scatter in the table : %8f\n",scatter_table);
fprintf(fp,"##### Scatter in the collimator: %8f\n",scatter_collim);
fprintf(fp,"##### Scatter in the crystal : %8f\n",scatter_crystal);
fprintf(fp,"##### Scatter in the backcompartment :  %8f\n",scatter_compartment);
fprintf(fp,"##### Scatter order 1: %8f\n",order1);
fprintf(fp,"##### Scatter order 2 : %8f\n",order2);
fprintf(fp,"##### Scatter order 3 : %8f\n",order3);
fprintf(fp,"##### Scatter order 4 : %8f\n",order4);
fprintf(fp,"##### Scatter order >4 : %8f",ordersup);

fclose(fp);
}

