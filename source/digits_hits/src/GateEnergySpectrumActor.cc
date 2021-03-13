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
  edepEvent = 0.;
  
  
  mSaveAsTextFlag = true;
  mSaveAsDiscreteSpectrumTextFlag = false;
  
  mEnableLETSpectrumFlag = false;
  mEnableLETFluenceSpectrumFlag = false;
  mEnableLETtoMaterialFluenceSpectrumFlag = false;
  mEnableQSpectrumFlag = false;
  mEnableEnergySpectrumNbPartFlag = false;
  mEnableEnergySpectrumFluenceCosFlag = false;
  mEnableEnergySpectrumFluenceTrackFlag = true;
  
  mEnableEnergySpectrumEdepFlag = true;
  
  mEnableEdepHistoFlag = false;
  mEnableEdepTimeHistoFlag = false;
  mEnableEdepTrackHistoFlag = false;
  mEnableEdepStepHistoFlag = false;
  mEnableElossHistoFlag = false;
  
  mEnableLogBinning = false;  
  mEnableEnergyPerUnitMass = false;
  mEnableRelativePrimEvents = false;
  mOtherMaterial = "G4_WATER";
  
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
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------



TH1D* GateEnergySpectrumActor::FactoryTH1D(const char *name, const char *title, Int_t nbinsx, Double_t xlow, Double_t xup, const char *xtitle, const char *ytitle)
{
    TH1D* th1d = new TH1D(name, title, nbinsx, xlow, xup);
    th1d->SetXTitle(xtitle);  
    th1d->SetYTitle(ytitle);
    return th1d;
}

TH1D* GateEnergySpectrumActor::FactoryTH1D2(const char *name, const char *title, const char *xtitle, const char *ytitle, double* binV, int nbins)
{
    //G4cout<< "nbins: " << nbins << G4endl;
    //G4cout << "============="<<G4endl<<G4endl;
    TH1D* th1d = new TH1D(name, title, nbins, binV);
    th1d->SetXTitle(xtitle);  
    th1d->SetYTitle(ytitle);
    return th1d;
}


double* GateEnergySpectrumActor::CreateBinVector(double emin, double emax, int nbins, bool enableLogBin)
{
    int nedges = nbins + 1;
    double* binV = new double[nedges];
    double dEn;
    //G4cout << "E: min, max " << emin<< ", "<<emax<< "; nbins " << nbins<< G4endl;
    if (enableLogBin){
      G4double eminLog = emin/MeV; // MeV is the default unit for energy in this actor
      if (eminLog < 0.000001) eminLog = 0.000001;
      double energyLogBinV_LB = TMath::Log10(eminLog/MeV);
      double energyLogBinV_UB = TMath::Log10(emax/MeV); 
      dEn = (energyLogBinV_UB - energyLogBinV_LB)/((double)nbins );
      for (int j = 0; j < nedges; j++) {
          double linSpacedV = energyLogBinV_LB + (double)j   * dEn;
          binV[j] = TMath::Power(10,  linSpacedV);
      }
      //G4cout<<"Log bin ind 0 1 and end " << binV[0]<< "; "<< binV[1]<< "; " << binV[nedges-1]<<G4endl;
    }
    else{
      dEn = (emax - emin)/((double)nbins );
      for (int j = 0; j < nedges; j++) {
          double linSpacedV = emin + (double)j   * dEn;
          binV[j] =  linSpacedV;
      }
      //G4cout<<"den: " << dEn << G4endl; 
      //G4cout<<"Lin bin ind 0 1 and end " << binV[0]<< "; "<< binV[1]<< "; " << binV[nedges-1]<<G4endl;
    }
    
    return binV;
}

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
  
  // Find G4_WATER.
  G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");
  // Find OtherMaterial
  G4NistManager::Instance()->FindOrBuildMaterial(mOtherMaterial);

  
  if (mEnableLETtoMaterialFluenceSpectrumFlag) {
      mEnableLETFluenceSpectrumFlag = true;
  }
      
  pTfile = new TFile(mSaveFilename,"RECREATE");

      if (mEnableEnergySpectrumNbPartFlag){
          
         pEnergySpectrumNbPart = this->FactoryTH1D2(
         "energySpectrumNbPart",
         "Energy Spectrum Number of particles", 
         "Energy (MeV)",
         "Number of particles",
          CreateBinVector(GetEmin() ,GetEmax(), GetENBins(), mEnableLogBinning), GetENBins());
          allEnabledTH1DHistograms.push_back(pEnergySpectrumNbPart);
      }
      if (mEnableEnergySpectrumFluenceCosFlag){
          
          pEnergySpectrumFluenceCos = this->FactoryTH1D2(
          "energySpectrumFluenceCos",
          "Energy Spectrum fluence 1/cos",
          "Energy (MeV)",
          "Fluence * Area [1]" ,
          CreateBinVector(GetEmin() ,GetEmax(), GetENBins(), mEnableLogBinning), GetENBins() );
          allEnabledTH1DHistograms.push_back(pEnergySpectrumFluenceCos);
      }

      if (mEnableEnergySpectrumFluenceTrackFlag){
          pEnergySpectrumFluenceTrack = this->FactoryTH1D2(
          "energySpectrumFluenceTrack",
          "Energy Spectrum fluence track",
          "Energy (MeV)",
          "Fluence * Area [1]" ,
          CreateBinVector(GetEmin() ,GetEmax(), GetENBins(), mEnableLogBinning), GetENBins() );
          allEnabledTH1DHistograms.push_back(pEnergySpectrumFluenceTrack);
      } 
        
      if (mEnableEnergySpectrumEdepFlag){
          pEnergyEdepSpectrum = this->FactoryTH1D2(
          "energyEdepSpectrum",
          "Energy Spectrum Edep",
          "Energy (MeV)",
          "Energy deposition (MeV)" ,
          CreateBinVector(GetEmin() ,GetEmax(), GetENBins(), mEnableLogBinning), GetENBins() );
          allEnabledTH1DHistograms.push_back(pEnergyEdepSpectrum);
          
      } 
      if (mEnableEdepHistoFlag){
          pEdep = this->FactoryTH1D2(
          "edepHisto",
          "Energy deposited per event",
          "Energy deposition (MeV)" ,
          "Frequency",
          CreateBinVector(GetEdepmin() ,GetEdepmax(), GetEdepNBins(),mEnableLogBinning) , GetEdepNBins());
          allEnabledTH1DHistograms.push_back(pEdep);
      } 
      if (mEnableEdepTrackHistoFlag){
          pEdepTrack = this->FactoryTH1D2(
          "edepTrackHisto",
          "Energy deposited per track",
          "Energy deposition (MeV)" ,
          "Frequency",
          CreateBinVector(GetEdepmin() ,GetEdepmax(), GetEdepNBins(), mEnableLogBinning), GetEdepNBins() );
          allEnabledTH1DHistograms.push_back(pEdepTrack);
          }       
     if (mEnableEdepStepHistoFlag){
          pEdepStep = this->FactoryTH1D2(
          "edepStepHisto",
          "Energy deposited per step",
          "Energy deposition (MeV)" ,
          "Frequency",
          CreateBinVector(GetEdepmin() ,GetEdepmax(), GetEdepNBins(), mEnableLogBinning), GetEdepNBins() );
          allEnabledTH1DHistograms.push_back(pEdepStep);
          } 
         

  if (mEnableLETSpectrumFlag) {
          pLETSpectrum = this->FactoryTH1D2(
          "LETSpectrum",
          "LET Spectrum",
          "LET (keV/um)" ,
          "Energy deposition (MeV)",
          CreateBinVector(GetLETmin() ,GetLETmax(), GetNLETBins(), mEnableLogBinning) , GetNLETBins());
          allEnabledTH1DHistograms.push_back(pLETSpectrum);
           
    }
  if (mEnableLETFluenceSpectrumFlag) {
          pLETFluenceSpectrum = this->FactoryTH1D2(
          "LETFluenceSpectrum",
          "LET Fluence Spectrum",
          "LET (keV/um)" ,
          "Fluence * Volume [mm]",
          CreateBinVector(GetLETmin() ,GetLETmax(), GetNLETBins(), mEnableLogBinning), GetNLETBins() );
          allEnabledTH1DHistograms.push_back(pLETFluenceSpectrum);
      } 
       
  if (mEnableLETtoMaterialFluenceSpectrumFlag) {
      std::string histBranchName = "";
      std::string materialToConvertToName = mOtherMaterial;
      histBranchName = "LETto" + materialToConvertToName + "FluenceSpectrum" ; 
      std::string histTitle = "LET to " + materialToConvertToName + " Fluence Spectrum";
          pLETtoMaterialFluenceSpectrum = this->FactoryTH1D2(
          histBranchName.c_str(),
          histTitle.c_str(),
          "LET to other material (keV/um)" ,
          "Fluence * Volume [mm]",
          CreateBinVector(GetLETmin() ,GetLETmax(), GetNLETBins(), mEnableLogBinning), GetNLETBins() );
          allEnabledTH1DHistograms.push_back(pLETtoMaterialFluenceSpectrum);
  }
  if (mEnableQSpectrumFlag) {
          pQSpectrum = this->FactoryTH1D2(
          "QSpectrum",
          "Q Spectrum",
          "Q (qq/MeV)" ,
          "Energy Deposition (MeV)",
          CreateBinVector(GetQmin() ,GetQmax(), GetNQBins(), mEnableLogBinning), GetNQBins() );
          allEnabledTH1DHistograms.push_back(pQSpectrum);
  }
  

  if (mEnableEdepTimeHistoFlag){
      pEdepTime  = new TH2D("edepHistoTime","Energy deposited with time per event",
                            GetEdepNBins(),0,20,GetEdepNBins(),GetEdepmin(),GetEdepmax());
      pEdepTime->SetXTitle("t (ns)");
      pEdepTime->SetYTitle("E_{dep} (MeV)");
  } 

  if (mEnableElossHistoFlag){
     pDeltaEc = this->FactoryTH1D2(
          "eLossHisto",
          "Energy loss",
          "E_{loss} (MeV)" ,
          "n.a.",
          CreateBinVector(GetEdepmin() ,GetEdepmax(), GetEdepNBins(), mEnableLogBinning) , GetEdepNBins());
          allEnabledTH1DHistograms.push_back(pDeltaEc);
   } 
  ResetData();
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
/// Save data
void GateEnergySpectrumActor::SaveData()
{
    double scaleFactor = 1.0;
   if (mEnableRelativePrimEvents){
       scaleFactor = nEvent;
    }
    for(std::list<TH1D*>::iterator it=allEnabledTH1DHistograms.begin();it!=allEnabledTH1DHistograms.end();++it)
      {
          (*it)->Scale(1./scaleFactor);
      }
   
  
  GateVActor::SaveData();
  pTfile->Write();
  
   //Also output data as txt if enabled
  if (mSaveAsTextFlag) {
      for(std::list<TH1D*>::iterator it=allEnabledTH1DHistograms.begin();it!=allEnabledTH1DHistograms.end();++it)
      {
          SaveAsText((*it),mSaveFilename);
      }
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
  edepEvent = 0.;
  tof  = 0;
   //G4cout<<"======================================"<<G4endl;
   //G4cout<<"=========== Pre Event Action ========="<<G4endl;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//void GateEnergySpectrumActor::EndOfEventAction(const G4Event* evH)
void GateEnergySpectrumActor::EndOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- End of Event\n");
  
    if (edepEvent > 0) {
        if (mEnableEdepHistoFlag){
            pEdep->Fill(edepEvent/MeV, 1);
              //G4cout<<"--------------Post Event Action ------------"<<G4endl;
              ////G4cout<<"Particle Name: " << evH->GetUserInformation()->print() <<G4endl;
              //G4cout<<"EdepEvent [eV]: " << edepEvent/eV<<G4endl;
              //G4cout<<"=========== End Event  ==============="<<G4endl;
              //G4cout<<"======================================"<<G4endl;
        }
        if (mEnableEdepTimeHistoFlag){
            pEdepTime->Fill(tof/ns,edepEvent/MeV);
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
  //G4cout<<"--------------- Pre User Tracking Action ---------------"<<G4endl;
  //G4cout<<"Particle Name: " << t->GetParticleDefinition()->GetParticleName() <<G4endl;
  //G4cout<<"EdepTrack [eV]: " << edepTrack/eV<<G4endl;
  //G4cout<< "-------------------------------------------------"<<G4endl;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::PostUserTrackingAction(const GateVVolume *, const G4Track* t)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- End of Track\n");
  double eloss = Ei-Ef;
  if (mEnableElossHistoFlag && eloss > 0) pDeltaEc->Fill(eloss/MeV,t->GetWeight() );
  
  if (mEnableEdepTrackHistoFlag && edepTrack > 0)  {
      pEdepTrack->Fill(edepTrack/MeV,t->GetWeight() );
  //G4cout<<"--------------- Post User Tracking Action ---------------"<<G4endl;
  //G4cout<<"Particle Name: " << t->GetParticleDefinition()->GetParticleName() <<G4endl;
  //G4cout<<"EdepTrack [eV]: " << edepTrack/eV<<G4endl;
  //G4cout<< "-------------------------------------------------"<<G4endl;
  //G4cout<< << <<G4endl;
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::UserSteppingAction(const GateVVolume *, const G4Step* step)
{
  assert(step->GetTrack()->GetWeight() == 1.); // edep doesnt handle weight

  if(step->GetTotalEnergyDeposit()>0.01) sumM1+=step->GetTotalEnergyDeposit();
  else if(step->GetTotalEnergyDeposit()>0.00001) sumM2+=step->GetTotalEnergyDeposit();
  else sumM3+=step->GetTotalEnergyDeposit();
  G4double edep = step->GetTotalEnergyDeposit();
  edepEvent += edep;
  
  //G4cout<<"  Edep [eV]: " << edep/eV<<G4endl;
  edepTrack += edep;
    if (mEnableEdepStepHistoFlag && edep > 0)  {
      pEdepStep->Fill(edep/MeV );
  //G4cout<<"--------------- Post User Tracking Action ---------------"<<G4endl;
  //G4cout<<"Particle Name: " << t->GetParticleDefinition()->GetParticleName() <<G4endl;
  //G4cout<<"EdepTrack [eV]: " << edepTrack/eV<<G4endl;
  //G4cout<< "-------------------------------------------------"<<G4endl;
  //G4cout<< << <<G4endl;
    }
  

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
  Ei=step->GetPreStepPoint()->GetKineticEnergy();
  
  if(newTrack){
    

    
    if (mEnableEnergySpectrumNbPartFlag){
        pEnergySpectrumNbPart->Fill(Ei/MeV/atomicMassScaleFactor,step->GetTrack()->GetWeight());
    }
    
    G4ThreeVector momentumDir = step->GetTrack()->GetMomentumDirection(); 
    
    if (mEnableEnergySpectrumFluenceCosFlag){
        double dz = TMath::Abs( momentumDir.z());
        if (dz > 0){
            //double Emean = (Ei+Ef)/2/MeV;
            double invAngle = 1/dz;
            pEnergySpectrumFluenceCos->Fill(Ei/MeV/atomicMassScaleFactor,step->GetTrack()->GetWeight()*invAngle);
        }
    }
    // uncommented A.Resch 30.Nov 2018
    //if (mSaveAsDiscreteSpectrumTextFlag) {
      //mDiscreteSpectrum.Fill(Ei/MeV, step->GetTrack()->GetWeight());
    //}
    newTrack=false;
  }
  G4double stepLength = step->GetStepLength();
   if (mEnableEnergySpectrumFluenceTrackFlag){
       
       pEnergySpectrumFluenceTrack->Fill(Ei/MeV/atomicMassScaleFactor,step->GetTrack()->GetWeight()*stepLength/mm);
       
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
  if(mEnableLETFluenceSpectrumFlag) {
      G4Material* material = step->GetPreStepPoint()->GetMaterial();//->GetName(); 
      G4double energyPre = step->GetPreStepPoint()->GetKineticEnergy();
      G4double energyPost = step->GetPostStepPoint()->GetKineticEnergy();
      G4double energyMean=(energyPre+energyPost)/2;
      G4ParticleDefinition* partdef = step->GetTrack()->GetDefinition();//->GetParticleName();
      G4double dedx;
      dedx = emcalc->ComputeElectronicDEDX(energyMean, partdef, material);
      pLETFluenceSpectrum->Fill(dedx/(keV/um),step->GetTrack()->GetWeight()*stepLength/mm);
            
     if(mEnableLETtoMaterialFluenceSpectrumFlag) {
          
           //other material
          static G4Material* OtherMaterial = G4Material::GetMaterial(mOtherMaterial,true);
          //// DISPLAY parameters of particles having DEDX=0
          //// Mainly gamma and neutron
          //DEDX = emcalc->ComputeTotalDEDX(energy, p, current_material, cut);
          dedx = emcalc->ComputeTotalDEDX(energyMean,step->GetTrack()->GetParticleDefinition(), OtherMaterial);
          pLETtoMaterialFluenceSpectrum->Fill(dedx/(keV/um),step->GetTrack()->GetWeight()*stepLength/mm);
     }
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
        << "# Other lines : 2 columns. 1) energy 2) Bin Width 2) frequency (nb divided by NbEvent)" << std::endl
        << "# Number of bins = " << histo->GetNbinsX() << std::endl
        << "# Content below the first bin: " << histo->GetBinContent(0) << std::endl
        << "# Content above the last  bin: " << histo->GetBinContent(histo->GetNbinsX()+2) << std::endl
        << "# Number of events: " << nEvent << std::endl
        << "2 " << histo->GetBinLowEdge(1) << std::endl; // start at 1
    for(int i=histo->GetXaxis()->GetFirst(); i< histo->GetXaxis()->GetLast(); i++) {
       oss << histo->GetBinCenter(i) << " " << histo->GetBinWidth(i)    << " " << histo->GetBinContent(i) << std::endl;  
    }
    oss.close();
  }
}
//-----------------------------------------------------------------------------

#endif
