/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateTEPCActor.hh"
#ifdef G4ANALYSIS_USE_ROOT

#include "GateTEPCActorMessenger.hh"
#include "GateMiscFunctions.hh"
#include "GateObjectStore.hh"
#include "GateSphere.hh"

#include "G4NistManager.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateTEPCActor::GateTEPCActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateTEPCActor() -- begin\n");

  // Default command setting
  mELogscale = true;
  mEmin = 0.01 * keV;
  mEmax = 1000 * keV;
  mEBinNumber = 150;
  mENOrders = 6;
  mNormByEvent = false;
  mSaveAsText = false;
  
  newEvent = true;
  
  pMessenger = new GateTEPCActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateTEPCActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateTEPCActor::~GateTEPCActor()
{
  GateDebugMessageInc("Actor",4,"~GateTEPCActor() -- begin\n");
  GateDebugMessageDec("Actor",4,"~GateTEPCActor() -- end\n");
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct
void GateTEPCActor::Construct()
{
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);
  EnableEndOfEventAction(true); // for save every n

  // Initialize ROOT histogram
  if(mEBinNumber < 1) { GateError("Error in " << GetName() << ": energy spectrum bin number is < 1"); }

  pTfile = new TFile(mSaveFilename,"RECREATE");
  
  mEmin /= keV;
  mEmax /= keV;
  
  if(mELogscale)
  {
    if(mEmin <= 0.0) { GateError("Error in " << GetName() << ": When using logscale, Emin value must be > 0"); }

    // Initialize bin width according to a log scale
    double *binEnergy = new double[mEBinNumber+1];
    double deltaE = (double)mENOrders / (double)mEBinNumber;
    double log10 = log(10.0);
    for(int i=0; i<mEBinNumber+1; i++)
    {
      binEnergy[i] = mEmin * exp(log10 * i * deltaE);
    }
    pLETspectrum = new TH1D("f","f(y) distribution", mEBinNumber, binEnergy);
    delete [] binEnergy;
  }
  else
  {
    // Initialize bin width according to a linear scale
    pLETspectrum = new TH1D("f","f(y) distribution", mEBinNumber, mEmin, mEmax);
  }

  // ROOT histogram - axis titles - options
  //pLETspectrum->Sumw2();
  pLETspectrum->SetXTitle("y (keV/#mum)");
  pLETspectrum->SetYTitle("f(y)");
  ResetData();
  
  // Calculation of the effective chord of the TEPC
  // WARNING : the attached volume MUST be a GateSphere
  GateMessage("Actor",0, "WARNING: For now, only spherical volumes are compatible with the TEPCactor. Please check that the attached volume is a 'sphere'." << G4endl);
  effectiveChord = (2.0 / 3.0) * 2.0 * ((GateSphere *)GetVolume())->GetSphereRmax()  * GetVolume()->GetMaterial()->GetDensity() / (g/cm3); // tested by A.Resch 29thOct2020: works and is correct
  G4cout<<"Effective chord length mm: " << effectiveChord<<G4endl;
  G4cout<<"Effective chord length um: "<< (effectiveChord / um)<<G4endl;
  G4cout<<"Radius Sphere: " << ((GateSphere *)GetVolume())->GetSphereRmax() <<G4endl;
  G4cout<<"Density: " << GetVolume()->GetMaterial()->GetDensity()/ (g/cm3) <<G4endl;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Copy the current material in a new one with the specified pressure
void GateTEPCActor::BuildMaterial(double pressure)
{
  mPressure = pressure;
  const G4Material *material = GetVolume()->GetMaterial();

  // new density
  double density = material->GetDensity() * (pressure / material->GetPressure());
  
  // new material name
  std::ostringstream oss;
  oss << pressure / pascal;
  G4String newMaterial = material->GetName() + "_" + oss.str() + "Pa";
  
  // create the new material, if it doesn't already exists
  if((G4NistManager::Instance()->FindOrBuildMaterial(newMaterial)) == NULL)
  {
    G4NistManager::Instance()->BuildMaterialWithNewDensity(newMaterial,material->GetName(),density);
  }
  
  // assign the new material to the attached volume
  GetVolume()->SetMaterialName(newMaterial);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateTEPCActor::SaveData()
{
  GateVActor::SaveData();

  // save ROOT histogram
  double normFactor = 1.0;
  if(mNormByEvent) { normFactor /= (double)eventNumber; }
  pLETspectrum->Scale(normFactor, "width");
  pTfile->Write();
  
  // save .txt file
  if(mSaveAsText)
  {
    // Compute new filename: remove extension, add name of the histo, add txt extension
    std::string filename = removeExtension(mSaveFilename);
    filename = filename + ".txt";

    // Create output file
    std::ofstream oss;
    OpenFileOutput(filename, oss);

    // write as text file with header and 3 columns: 1) energy[keV] 2) binWidth[keV] 3) frequency
    oss << "energy[keV] binWidth[keV] frequency" << std::endl;
    int binFirst = pLETspectrum->GetXaxis()->GetFirst();
    int binLast  = pLETspectrum->GetXaxis()->GetLast();
    for(int i=binFirst; i<=binLast; i++)
    {
      oss << pLETspectrum->GetBinCenter(i) << " " << pLETspectrum->GetBinWidth(i) << " " << pLETspectrum->GetBinContent(i) << std::endl;
    }
    oss.close();
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTEPCActor::ResetData()
{
  pLETspectrum->Reset();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTEPCActor::BeginOfRunAction(const G4Run *)
{
  GateDebugMessage("Actor", 3, "GateTEPCActor -- Begin of Run\n");

  eventNumber = 0;
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTEPCActor::BeginOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateTEPCActor -- Begin of Event\n");
  
  eventNumber++;
  newEvent = true;
  edepByEvent = 0.0;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// record the linear energy transfer of the current event
void GateTEPCActor::EndOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateTEPCActor -- End of Event\n");
    //myEvent->GetUserInformation();
    //myEvent->GetPrimaryVertex()->Print();
    
  double y = (edepByEvent / keV) / (effectiveChord / um);
  if(edepByEvent > 0.0) { pLETspectrum->Fill(y, 1.0); }
  
//  if((eventNumber % 1000) == 0) { G4cout << "number of event -- " << eventNumber << G4endl; }
}
//-----------------------------------------------------------------------------

 
//-----------------------------------------------------------------------------
void GateTEPCActor::PreUserTrackingAction(const GateVVolume *, const G4Track *)
{
  GateDebugMessage("Actor", 3, "GateTEPCActor -- Begin of Track\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTEPCActor::PostUserTrackingAction(const GateVVolume *, const G4Track *)
{
  GateDebugMessage("Actor", 3, "GateTEPCActor -- End of Track\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTEPCActor::UserSteppingAction(const GateVVolume *, const G4Step *step)
{
  edepByEvent += step->GetTotalEnergyDeposit();
}
//-----------------------------------------------------------------------------


#endif
