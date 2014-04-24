//ooooooooooOOOOOOO00000000000OOOOOOOOOOOooooooooooooo//
//
//ooooooooooOOOOOOO00000000000OOOOOOOOOOOooooooooooooo//

{
 gROOT->Reset();

 int nCol=1;      // n.column
 int nRow=1;      // n.row
 int nZ=400;        // n.z
 int imBitNumber=32;


 int              nPoint_tot = nCol*nRow*nZ;
 long             lSize = nPoint_tot*(imBitNumber/8);

 Double_t x1[nZ], y1[nZ];
 Double_t x2[nZ], y2[nZ];


//... define the buffer
  float       *buffer_p;
  float       *buffer_g;

  buffer_p = (float*) malloc (lSize);
  buffer_g = (float*) malloc (lSize);

//... open the image
  FILE* protonFile;
  FILE* gammaFile;

  protonFile = fopen ("output/Config-proton-Edep.img" , "rb" );
  gammaFile = fopen ("output/Config-gamma-Edep.img" , "rb" );

//... read the image
 fread (buffer_p,1,lSize,protonFile);
 fread (buffer_g,1,lSize,gammaFile);


 for(int i = 0 ; i<nZ ; i++){
     x1[i]  = i;
     y1[i] = buffer_p[i];

     x2[i]  = i;
     y2[i] = buffer_g[i];
 }

fclose(protonFile);
fclose(gammaFile);


   TGraph *gr1 = new TGraph(nZ,x1,y1);
   gr1->SetLineColor(1);
   gr1->SetLineWidth(3);
   gr1->SetTitle("BenchRT energy curves");
   gr1->GetHistogram()->SetXTitle("Depth - mm");
   gr1->GetHistogram()->SetYTitle("Energy - MeV");
   gr1->GetXaxis()->SetTitleOffset(1.1);
   gr1->GetYaxis()->SetTitleOffset(1.35);

   TGraph *gr2 = new TGraph(nZ,x2,y2);
   gr2->SetLineColor(2);
   gr2->SetLineWidth(3);

   TLegend *leg = new TLegend(0.5,0.5,0.8,0.65);
	leg->SetFillColor(0);
	leg->SetTextSize(0.03);
	leg->AddEntry(gr1,"Proton beam: 150 MeV ","lp");
	leg->AddEntry(gr2,"Photon beam: 18 MeV","lp");


   TCanvas *c1 = new TCanvas("c1","transparent pad",200,10,700,500);

   TPad *pad1 = new TPad("pad1","",0,0,1,1);
   TPad *pad2 = new TPad("pad2","",0,0,1,1);
   pad2->SetFillStyle(4000); //will be transparent
   pad1->SetGrid();

   pad1->Draw();
   pad1->cd();

   gr1->Draw("apc");
   pad1->Update(); //this will force the generation of the "stats" box
   leg->Draw();
   c1->cd();

   //compute the pad range with suitable margins
   Double_t ymin = 0;
   Double_t ymax = 18000;
   Double_t dy = (ymax-ymin)/0.8; //10 per cent margins top and bottom
   Double_t xmin = 0;
   Double_t xmax = 400;
   Double_t dx = (xmax-xmin)/0.8; //10 per cent margins left and right
   pad2->Range(xmin-0.1*dx,ymin-0.1*dy,xmax+0.1*dx,ymax+0.1*dy);
   pad2->Draw();
   pad2->cd();

   gr2->Draw("spc");

   pad2->Update();
   leg->Draw();
   pad2->Modified();

   TGaxis *axis = new TGaxis(xmax,ymin,xmax,ymax,ymin,ymax,50510,"+L");
   axis->SetLabelColor(kRed);
   axis->Draw();
   c1->SaveAs("output/benchmarkRT.gif");
}

//ooooooooooOOOOOOO00000000000OOOOOOOOOOOooooooooooooo//
