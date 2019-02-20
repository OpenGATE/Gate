/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateEnergySpectrumActor.hh"
#ifdef G4ANALYSIS_USE_ROOT

#include "GateEnergySpectrumActorMessenger.hh"
#include "GateMiscFunctions.hh"

// g4 // inserted 30 Jan 2016:
#include <G4EmCalculator.hh>
#include <G4VoxelLimits.hh>
#include <G4NistManager.hh>
#include <G4PhysicalConstants.hh>



//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateEnergySpectrumActor::GateEnergySpectrumActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateEnergySpectrumActor() -- begin\n");

  mEmin = 0.;
  mEmax = 50.;
  mENBins = 100;
  
  mLETmin = 0.;
  mLETmax = 100.;
  mLETBins = 1000;

  mQmin = 0.;
  mQmax = 10.;
  mQBins = 2000;

  mEdepmin = 0.;
  mEdepmax = 50.;
  mEdepNBins = 100;

  Ei = 0.;
  Ef = 0.;
  newEvt = true;
  newTrack = true;
  sumNi=0.;
  nTrack=0;
  sumM1=0.;
  sumM2=0.;
  sumM3=0.;
  edep = 0.;
  
  mSaveAsTextFlag = true;
  mSaveAsDiscreteSpectrumTextFlag = false;
  
  mEnableLETSpectrumFlag = false;
  mEnableQSpectrumFlag = false;
  mEnableEnergySpectrumNbPartFlag = false;
  mEnableEnergySpectrumFluenceCosFlag = false;
  mEnableEnergySpectrumFluenceTrackFlag = true;
  
  mEnableEnergySpectrumEdepFlag = true;
  
  mEnableEdepHistoFlag = false;
  mEnableEdepTimeHistoFlag = false;
  mEnableEdepTrackHistoFlag = false;
  mEnableElossHistoFlag = false;
  
  mEnableLogBinning = false;  
  mEnableEnergyPerUnitMass = false;
  
  emcalc = new G4EmCalculator;

  pMessenger = new GateEnergySpectrumActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateEnergySpectrumActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateEnergySpectrumActor::~GateEnergySpectrumActor()
{
  GateDebugMessageInc("Actor",4,"~GateEnergySpectrumActor() -- begin\n");
  GateDebugMessageDec("Actor",4,"~GateEnergySpectrumActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct
void GateEnergySpectrumActor::Construct()
{
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);
  EnableEndOfEventAction(true); // for save every n
  

  
  if (mEnableLogBinning){
      G4double mEminLog = mEmin/MeV; // MeV is the default unit for energy in this actor
      if (mEminLog < 0.000001) mEminLog = 0.000001;
      double energyLogBinV_LB = TMath::Log10(mEminLog/MeV);
      double energyLogBinV_UB = TMath::Log10(mEmax/MeV); 
      dEn = (energyLogBinV_UB - energyLogBinV_LB)/((double)mENBins );
      eBinV = new double[mENBins+1];
      for (int j = 0; j < (mENBins+1); j++) {
          
          double linSpacedV = energyLogBinV_LB + (double)j   * dEn;
          eBinV[j] = TMath::Power(10,  linSpacedV);
      }
  }
  
  pTfile = new TFile(mSaveFilename,"RECREATE");
  if (!mEnableLogBinning){
      if (mEnableEnergySpectrumNbPartFlag){
          pEnergySpectrumNbPart = new TH1D("energySpectrumNbPart","Energy Spectrum Number of particles",GetENBins(),GetEmin() ,GetEmax() );
          pEnergySpectrumNbPart->SetXTitle("Energy (MeV)");  
          pEnergySpectrumNbPart->SetYTitle("Number of particles");
          allEnabledTH1DHistograms.push_back(pEnergySpectrumNbPart);
      }
      if (mEnableEnergySpectrumFluenceCosFlag){
          pEnergySpectrumFluenceCos = new TH1D("energySpectrumFluenceCos","Energy Spectrum fluence 1/cos",GetENBins(),GetEmin() ,GetEmax() );
          pEnergySpectrumFluenceCos->SetXTitle("Energy (MeV)");  
          pEnergySpectrumFluenceCos->SetYTitle("Fluence [cm^-2]");
          allEnabledTH1DHistograms.push_back(pEnergySpectrumFluenceCos);
      }

      if (mEnableEnergySpectrumFluenceTrackFlag){
          pEnergySpectrumFluenceTrack = new TH1D("energySpectrumFluenceTrack","Energy Spectrum fluence Track",GetENBins(),GetEmin() ,GetEmax() );
          pEnergySpectrumFluenceTrack->SetXTitle("Energy (MeV)");
          pEnergySpectrumFluenceTrack->SetYTitle("Fluence * Volume [cm]");
          allEnabledTH1DHistograms.push_back(pEnergySpectrumFluenceTrack);
      } 
        
      if (mEnableEnergySpectrumEdepFlag){
          pEnergyEdepSpectrum = new TH1D("energyEdepSpectrum","Energy Edep Spectrum",GetENBins(),GetEmin() ,GetEmax() );
          pEnergyEdepSpectrum->SetXTitle("Energy (MeV)");
          pEnergyEdepSpectrum->SetYTitle("Energy deposition (MeV)");
          allEnabledTH1DHistograms.push_back(pEnergyEdepSpectrum);
      } 
  }
  else
  {
      if (mEnableEnergySpectrumNbPartFlag){
          pEnergySpectrumNbPart = new TH1D("energySpectrumNbPart","Energy Spectrum Number of particles",mENBins,eBinV);
          pEnergySpectrumNbPart->SetXTitle("Energy (MeV)");  
          pEnergySpectrumNbPart->SetYTitle("Number of particles");
          allEnabledTH1DHistograms.push_back(pEnergySpectrumNbPart);
          }
      if (mEnableEnergySpectrumFluenceCosFlag){
          pEnergySpectrumFluenceCos = new TH1D("energySpectrumFluenceCos","Energy Spectrum fluence 1/cos", mENBins,eBinV );
          pEnergySpectrumFluenceCos->SetXTitle("Energy (MeV)");  
          pEnergySpectrumFluenceCos->SetYTitle("Fluence [cm^-2]");   
          allEnabledTH1DHistograms.push_back(pEnergySpectrumFluenceCos);   
          }
      if (mEnableEnergySpectrumFluenceTrackFlag){
          pEnergySpectrumFluenceTrack = new TH1D("energySpectrumFluenceTrack","Energy Spectrum fluence Track", mENBins,eBinV );
          pEnergySpectrumFluenceTrack->SetXTitle("Energy (MeV)");
          pEnergySpectrumFluenceTrack->SetYTitle("Fluence * Volume [cm]");
          allEnabledTH1DHistograms.push_back(pEnergySpectrumFluenceTrack);
          } 
      if (mEnableEnergySpectrumEdepFlag){
          pEnergyEdepSpectrum = new TH1D("energyEdepSpectrum","Energy Edep Spectrum", mENBins,eBinV);
          pEnergyEdepSpectrum->SetXTitle("Energy (MeV)");
          pEnergyEdepSpectrum->SetYTitle("Energy deposition (MeV)");
          allEnabledTH1DHistograms.push_back(pEnergyEdepSpectrum);
      } 
  }


  if (mEnableLETSpectrumFlag) {
      pLETSpectrum = new TH1D("LETSpectrum","LET Spectrum",GetNLETBins(),GetLETmin() ,GetLETmax() );
      pLETSpectrum->SetXTitle("LET (keV/um)");
      pLETSpectrum->SetYTitle("Energy deposition (MeV)");
      allEnabledTH1DHistograms.push_back(pLETSpectrum);
  }
  if (mEnableQSpectrumFlag) {
      pQSpectrum = new TH1D("QSpectrum","Q Spectrum",GetNQBins(),GetQmin() ,GetQmax() );
      pQSpectrum->SetXTitle("Q (qq/MeV)");
      pQSpectrum->SetYTitle("Energy deposition (MeV)");
      allEnabledTH1DHistograms.push_back(pQSpectrum);
  }
  
  if (mEnableEdepHistoFlag){
      pEdep  = new TH1D("edepHisto","Energy deposited per event",GetEdepNBins(),GetEdepmin() ,GetEdepmax() );
      pEdep->SetXTitle("E_{dep} (MeV)");
      allEnabledTH1DHistograms.push_back(pEdep);
  } 

  if (mEnableEdepTimeHistoFlag){
      pEdepTime  = new TH2D("edepHistoTime","Energy deposited with time per event",
                            GetEdepNBins(),0,20,GetEdepNBins(),GetEdepmin(),GetEdepmax());
      pEdepTime->SetXTitle("t (ns)");
      pEdepTime->SetYTitle("E_{dep} (MeV)");

  } 

  if (mEnableEdepTrackHistoFlag){
      pEdepTrack  = new TH1D("edepTrackHisto","Energy deposited per track",GetEdepNBins(),GetEdepmin() ,GetEdepmax() );
      pEdepTrack->SetXTitle("E_{dep} (MeV)");
      allEnabledTH1DHistograms.push_back(pEdepTrack);
  } 

  if (mEnableElossHistoFlag){
      pDeltaEc = new TH1D("eLossHisto","Energy loss",GetEdepNBins(),GetEdepmin() ,GetEdepmax() );
      pDeltaEc ->SetXTitle("E_{loss} (MeV)");
      allEnabledTH1DHistograms.push_back(pDeltaEc);
  } 
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateEnergySpectrumActor::SaveData()
{
    
  GateVActor::SaveData();
  pTfile->Write();
  
   //Also output data as txt if enabled
  if (mSaveAsTextFlag) {
      for(std::list<TH1D*>::iterator it=allEnabledTH1DHistograms.begin();it!=allEnabledTH1DHistograms.end();++it)
      {
          SaveAsText((*it),mSaveFilename);
      }
    ////SaveAsText(pEnergySpectrumNbPart, mSaveFilename);
    ////SaveAsText(pEnergySpectrumFluenceCos, mSaveFilename);
    //SaveAsText(pEnergySpectrumFluenceTrack, mSaveFilename);
    ////SaveAsText(pEnergyEdepSpectrum, mSaveFilename);
    ////SaveAsText(pEdep, mSaveFilename);
    ////// SaveAsText(pEdepTime, mSaveFilename); no TH2D
    ////SaveAsText(pEdepTrack, mSaveFilename);
    ////SaveAsText(pDeltaEc, mSaveFilename);
  }
  pTfile->Close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::ResetData()
{
    for(std::list<TH1D*>::iterator it=allEnabledTH1DHistograms.begin();it!=allEnabledTH1DHistograms.end();++it)
      {
          (*it)->Reset();
      }
  //if (mEnableEnergySpectrumNbPartFlag){
    //pEnergySpectrumNbPart->Reset();
  //}
  
  //if (mEnableEnergySpectrumFluenceCosFlag){
      //pEnergySpectrumFluenceCos->Reset();
  //}
  
  //if (mEnableEnergySpectrumFluenceTrackFlag){
      //pEnergySpectrumFluenceTrack->Reset();
  //}
  //if (mEnableEnergySpectrumEdepFlag){
      //pEnergyEdepSpectrum->Reset();
  //}
   
  //if (mEnableLETSpectrumFlag) {
      //pLETSpectrum->Reset();
  //}
  //if (mEnableQSpectrumFlag){
      //pQSpectrum->Reset();
  //}
  
  //if (mEnableEdepHistoFlag){
      //pEdep->Reset();
  //}
  // timeHisto is a 2D hist - others are 1D
  if (mEnableEdepTimeHistoFlag){
      pEdepTime->Reset();
  }
  
  //if (mEnableEdepTrackHistoFlag){
      //pEdepTrack->Reset();
  //}
  
  //if (mEnableElossHistoFlag){
      //pDeltaEc->Reset();
  //}
  nEvent = 0;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::BeginOfRunAction(const G4Run *)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- Begin of Run\n");
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::BeginOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- Begin of Event\n");
  newEvt = true;
  edep = 0.;
  tof  = 0;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::EndOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- End of Event\n");
  
    if (edep > 0) {
        if (mEnableEdepHistoFlag){
            pEdep->Fill(edep/MeV);
        }
        if (mEnableEdepTimeHistoFlag){
            pEdepTime->Fill(tof/ns,edep/MeV);
        }
      }
  
  
  
      
  nEvent++;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::PreUserTrackingAction(const GateVVolume *, const G4Track* t)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- Begin of Track\n");
  newTrack = true;
  if (t->GetParentID()==1) nTrack++;
  edepTrack = 0.;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::PostUserTrackingAction(const GateVVolume *, const G4Track* t)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- End of Track\n");
  double eloss = Ei-Ef;
  if (mEnableElossHistoFlag && eloss > 0) pDeltaEc->Fill(eloss/MeV,t->GetWeight() );
  
  if (mEnableEdepTrackHistoFlag && edepTrack > 0)  pEdepTrack->Fill(edepTrack/MeV,t->GetWeight() );
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::UserSteppingAction(const GateVVolume *, const G4Step* step)
{
  assert(step->GetTrack()->GetWeight() == 1.); // edep doesnt handle weight

  if(step->GetTotalEnergyDeposit()>0.01) sumM1+=step->GetTotalEnergyDeposit();
  else if(step->GetTotalEnergyDeposit()>0.00001) sumM2+=step->GetTotalEnergyDeposit();
  else sumM3+=step->GetTotalEnergyDeposit();

  edep += step->GetTotalEnergyDeposit();
  edepTrack += step->GetTotalEnergyDeposit();

  if (newEvt) {
    double pretof = step->GetPreStepPoint()->GetGlobalTime();
    double posttof = step->GetPostStepPoint()->GetGlobalTime();
    tof = pretof + posttof;
    tof /= 2;
    //cout << "****************** new event tof=" << pretof << "/" << posttof << "/" << tof << " edep=" << edep << endl;
    newEvt = false;
  } else {
    double pretof = step->GetPreStepPoint()->GetGlobalTime();
    double posttof = step->GetPostStepPoint()->GetGlobalTime();
    double ltof = pretof + posttof;
    ltof /= 2;
    //cout << "****************** diff tof=" << ltof << " edep=" << edep << endl;
  }
    G4double atomicMassScaleFactor = 1.;
    if (mEnableEnergyPerUnitMass){
        atomicMassScaleFactor = (double)(step->GetTrack()->GetParticleDefinition()->GetAtomicMass());
    }
  Ef=step->GetPostStepPoint()->GetKineticEnergy();
  if(newTrack){
    Ei=step->GetPreStepPoint()->GetKineticEnergy();

    
    if (mEnableEnergySpectrumNbPartFlag){
        pEnergySpectrumNbPart->Fill(Ei/MeV/atomicMassScaleFactor,step->GetTrack()->GetWeight());
    }
    
    G4ThreeVector momentumDir = step->GetTrack()->GetMomentumDirection(); 
    
    if (mEnableEnergySpectrumFluenceCosFlag){
        double dz = TMath::Abs( momentumDir.z());
        if (dz > 0){
            //double Emean = (Ei+Ef)/2/MeV;
            double invAngle = 1/dz;
            //if (invAngle > 10) invAngle = 10;
            pEnergySpectrumFluenceCos->Fill(Ei/MeV/atomicMassScaleFactor,step->GetTrack()->GetWeight()*invAngle);
        }
    }
    // uncommented A.Resch 30.Nov 2018
    //if (mSaveAsDiscreteSpectrumTextFlag) {
      //mDiscreteSpectrum.Fill(Ei/MeV, step->GetTrack()->GetWeight());
    //}
    newTrack=false;
  }
  
   if (mEnableEnergySpectrumFluenceTrackFlag){
       G4double stepLength = step->GetStepLength();
       pEnergySpectrumFluenceTrack->Fill(Ei/MeV/atomicMassScaleFactor,step->GetTrack()->GetWeight()*stepLength);
       
   }
   if (mEnableEnergySpectrumEdepFlag){
       pEnergyEdepSpectrum->Fill(Ei/MeV/atomicMassScaleFactor,step->GetTrack()->GetWeight()*step->GetTotalEnergyDeposit()/MeV);
   }
  if(mEnableLETSpectrumFlag) {
      G4Material* material = step->GetPreStepPoint()->GetMaterial();//->GetName(); 
      G4double energy1 = step->GetPreStepPoint()->GetKineticEnergy();
      G4double energy2 = step->GetPostStepPoint()->GetKineticEnergy();
      G4double energy=(energy1+energy2)/2;
      G4ParticleDefinition* partname = step->GetTrack()->GetDefinition();//->GetParticleName();
      G4double dedx;
      dedx = emcalc->ComputeElectronicDEDX(energy, partname, material);
      pLETSpectrum->Fill(dedx/(keV/um),step->GetTrack()->GetWeight()*edep/MeV);
            
  }  
  
  if(mEnableQSpectrumFlag) {
     
      G4double energy1Q = step->GetPreStepPoint()->GetKineticEnergy();
      G4double energy2Q = step->GetPostStepPoint()->GetKineticEnergy();
      G4double energyQ=(energy1Q+energy2Q)/2;
      G4int chargeQ = step->GetTrack()->GetDefinition()->GetAtomicNumber();
      
      G4double Q =chargeQ; // to convert Int to Double
      Q*=Q; // now chargeQ is squared
      Q/=(energyQ/MeV); // now we divide chargeQ^2 / energyQ
      pQSpectrum->Fill(Q,step->GetTrack()->GetWeight()*step->GetTotalEnergyDeposit()/MeV);
  }
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::SaveAsText(TH1D * histo, G4String initial_filename)
{
  // Compute new filename: remove extension, add name of the histo, add txt extension
  std::string filename = removeExtension(initial_filename);
  filename = filename + "_"+histo->GetName()+".txt";

  // Create output file
  std::ofstream oss;
  OpenFileOutput(filename, oss);

  // FIXME
  if (mSaveAsDiscreteSpectrumTextFlag) {
    oss << "# First line is two numbers " << std::endl
        << "#     First value is '1', it means 'discrete energy mode'" << std::endl
        << "#     Second value is ignored" << std::endl
        << "# Other lines : 2 columns. 1) energy 2) probability (nb divided by NbEvent)" << std::endl
        << "# Number of bins = " << mDiscreteSpectrum.size() << std::endl
        << "# Number of events: " << nEvent << std::endl
        << "1 0" << std::endl;
    for(int i=0; i<mDiscreteSpectrum.size(); i++) {
      oss << mDiscreteSpectrum.GetEnergy(i) << " " << mDiscreteSpectrum.GetValue(i) << std::endl; // removed division by nEvent for case nEvent = 0 ;
    }
    oss.close();
  }
  else {
    // write as text file with header and 2 columns: 1) energy 2) probability
    // The header is two numbers:
    ///    1 because it is mode 1 (see gps UserSpectrum)
    //     Emin of the histo

    // Root convention
    // For all histogram types: nbins, xlow, xup
    //         bin = 0;       underflow bin
    //         bin = 1;       first bin with low-edge xlow INCLUDED
    //         bin = nbins;   last bin with upper-edge xup EXCLUDED
    //         bin = nbins+1; overflow bin
    oss << "# First line is two numbers " << std::endl
        << "#     First value is '2', it means 'histogram mode'" << std::endl
        << "#     Second value is 'Emin' of the histogram" << std::endl
        << "# Other lines : 2 columns. 1) energy 2) probability (nb divided by NbEvent)" << std::endl
        << "# Number of bins = " << histo->GetNbinsX() << std::endl
        << "# Content below the first bin: " << histo->GetBinContent(0) << std::endl
        << "# Content above the last  bin: " << histo->GetBinContent(histo->GetNbinsX()+2) << std::endl
        << "# Content above the last  bin: " << histo->GetBinContent(histo->GetNbinsX()+2) << std::endl
        << "# Number of events: " << nEvent << std::endl
        << "2 " << histo->GetBinLowEdge(1) << std::endl; // start at 1
    for(int i=1; i<histo->GetNbinsX()+1; i++) {
      oss << histo->GetBinLowEdge(i) + histo->GetBinWidth(i) << " " << histo->GetBinContent(i) << std::endl; // removed division by nEvent for case nEvent = 0 ;
    }
    oss.close();
  }
}
//-----------------------------------------------------------------------------

#endif
