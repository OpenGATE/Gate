/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "G4SystemOfUnits.hh"

#include "GateVSourceVoxelReader.hh"
#include <iostream>
#include <fstream>
#include "GateVSource.hh"
#include "Randomize.hh"

#include "GateSourceVoxelLinearTranslator.hh"
#include "GateSourceVoxelRangeTranslator.hh"
#include "GateSourceMgr.hh"

GateVSourceVoxelReader::GateVSourceVoxelReader(GateVSource* source)
  : m_source(source)
  , m_voxelTranslator(0)
{
  m_position = G4ThreeVector();
  m_activityTotal = 0. * becquerel;
  m_activityMax   = 0. * becquerel;

  G4double voxelSize = 1.*mm;
  m_voxelSize = G4ThreeVector(voxelSize,voxelSize,voxelSize);
}

GateVSourceVoxelReader::~GateVSourceVoxelReader()
{
  if (m_voxelTranslator) {
    delete m_voxelTranslator;
  }
  m_sourceVoxelIntegratedActivities.clear();
}


void GateVSourceVoxelReader::Dump(G4int level) 
{

  G4cout << "  Voxel reader ----------> " << m_type
	 << "  number of voxels       : " << m_sourceVoxelActivities.size() << G4endl
	 << "  total activity (Bq)    : " << GetTotalActivity()/becquerel << G4endl
	 << "  position  (mm)         : " 
	 << GetPosition().x()/mm << " " 
	 << GetPosition().y()/mm << " " 
	 << GetPosition().z()/mm << G4endl
	 << "  voxel size  (mm)    : " 
	 << GetVoxelSize().x()/mm << " " 
	 << GetVoxelSize().y()/mm << " " 
	 << GetVoxelSize().z()/mm << G4endl;

  if (level > 2) {
    GateSourceActivityMap::iterator voxel;
    for (voxel = m_sourceVoxelActivities.begin(); voxel != m_sourceVoxelActivities.end(); voxel++) {
      G4cout << "   Index" 
	     << " " << (*voxel).first[0] 
	     << " " << (*voxel).first[1] 
	     << " " << (*voxel).first[2] 
	     << " Activity (Bq) " << (*voxel).second / becquerel << G4endl;
    }
  }
}

std::vector<G4int> GateVSourceVoxelReader::GetNextSource()
{
  // the method decides which is the source that has to be used for this event

  std::vector<G4int> firstSource;

  if (m_sourceVoxelActivities.size()==0) {
    G4cout << "GateVSourceVoxelReader::GetNextSource : WARNING: No source available" << G4endl;
    return firstSource;
  } else {
    // if there is at least one voxel

    // now assign the event to one voxel, according to the relative activity
    // integral method
    // from STL doc: iterator upper_bound(const key_type& k)   Sorted Associative Container   Finds the first element whose key greater than k.
    firstSource = (*(m_sourceVoxelIntegratedActivities.upper_bound(G4UniformRand() * m_activityTotal))).second;

  }

  if (nVerboseLevel>1) 
    G4cout << "GateVSourceVoxelReader::GetNextSource : source chosen : " 
	   << " " << firstSource[0] 
	   << " " << firstSource[1] 
	   << " " << firstSource[2] 
	   << G4endl;

  return firstSource;
}

void GateVSourceVoxelReader::AddVoxel(G4int ix, G4int iy, G4int iz, G4double activity)
{
  // this method is used by the ReadFile method. 
  // Note: The decision to create a new voxel has already been taken before.

  // create the vector key
  std::vector<G4int>* index = new std::vector<G4int>;
  index->push_back(ix);
  index->push_back(iy);
  index->push_back(iz);

  // if a source had already been inserted with the same key (ix,iy,iz) this will substitute the previous one; 
  // thus we must delete the previous from the sum of the activities and we recompute the maximum
  if (m_sourceVoxelActivities[*index] != 0) {
    G4cout << "GateVSourceVoxelReader::AddVoxel: already existing voxel, activity replaced" << G4endl;
    m_activityTotal -= m_sourceVoxelActivities[*index];

    // loop over all the voxels to recompute the maximum of the activies
    // without this voxel
    m_sourceVoxelActivities[*index] = 0. * becquerel;
    GateSourceActivityMap::iterator voxel;
    m_activityMax = 0. * becquerel;
    for (voxel = m_sourceVoxelActivities.begin(); voxel != m_sourceVoxelActivities.end(); voxel++) {
      G4double iterActivity = (*voxel).second;
      if (iterActivity > m_activityMax) {
	m_activityMax = iterActivity;
      }
    }

  }
  m_activityTotal += activity;

  
  if (activity > m_activityMax) {
    m_activityMax = activity;
  }

  m_sourceVoxelActivities[*index] = activity;
  
}
void GateVSourceVoxelReader::AddVoxel_FAST(G4int ix, G4int iy, G4int iz, G4double activity)
{ // no check if Voxel already existed to speed-up
  //
  // create the vector key
  std::vector<G4int> index;
  index.push_back(ix);
  index.push_back(iy);
  index.push_back(iz);


  m_sourceVoxelActivities[index] = activity;
  
}
void GateVSourceVoxelReader::InsertTranslator(G4String translatorType)
{
  if (m_voxelTranslator) {
    G4cout << "GateVSourceVoxelReader::InsertTranslator: voxel translator already defined" << G4endl;
  } else {
    if (translatorType == G4String("linear")) {
      m_voxelTranslator = new GateSourceVoxelLinearTranslator(this);
    } else if (translatorType == G4String("range")) {
      m_voxelTranslator = new GateSourceVoxelRangeTranslator(this);
    } else {
      G4cout << "GateVSourceVoxelReader::InsertTranslator: unknown translator type" << G4endl;
    }
  }

}

void GateVSourceVoxelReader::RemoveTranslator()
{
  if (m_voxelTranslator) {
    delete m_voxelTranslator;
    m_voxelTranslator = 0;
  } else {
    G4cout << "GateVSourceVoxelReader::RemoveTranslator: voxel translator not defined" << G4endl;
  }
}

void GateVSourceVoxelReader::PrepareIntegratedActivityMap()
{
  // erase all the elements of the old integrated activity map
  m_sourceVoxelIntegratedActivities.clear();

  // create the new integrated activity map
  m_activityTotal = 0.;
  GateSourceActivityMap::iterator voxel;
  for (voxel = m_sourceVoxelActivities.begin(); voxel != m_sourceVoxelActivities.end(); voxel++) {
    m_activityTotal += (*voxel).second;
    G4double* intActivityKey = new G4double(m_activityTotal);
    m_sourceVoxelIntegratedActivities[*intActivityKey] = (*voxel).first;
  
  delete intActivityKey;
  }

  if (nVerboseLevel>1) {
    G4int nVoxels = 0;
    GateSourceActivityMap::iterator voxel = m_sourceVoxelActivities.begin();
    GateSourceIntegratedActivityMap::iterator intVoxel;
    for (intVoxel = m_sourceVoxelIntegratedActivities.begin(); intVoxel != m_sourceVoxelIntegratedActivities.end(); intVoxel++, voxel++) {
      nVoxels++;
      G4cout << "[GateVSourceVoxelReader::PrepareIntegratedActivityMap] " 
	     << "   voxel: " << ((*voxel).first)[0] << " " << ((*voxel).first)[1] << " " << ((*voxel).first)[2] 
	     << "   activity : (Bq) " << ((*voxel).second) / becquerel 
	     << "   intVoxel: " << ((*intVoxel).second)[0] << " " << ((*intVoxel).second)[1] << " " << ((*intVoxel).second)[2] 
	     << "   integrated: (Bq) " << ((*intVoxel).first)  / becquerel << G4endl;
    }
  }
  m_tactivityTotal = m_activityTotal;  // added by I. Martinez-Rovira (immamartinez@gmail.com)
}

/* PY Descourt 08/09/2009 */
void GateVSourceVoxelReader::Initialize()
{
m_sourceVoxelActivities.clear();
}
G4double GateVSourceVoxelReader::GetTimeSampling()
{
return m_TS;
}
void GateVSourceVoxelReader::SetTimeSampling( G4double aTS )
{ m_TS = aTS; }


void GateVSourceVoxelReader::UpdateActivities(G4String  HFN, G4String FN )
{
  static G4bool IsFirstTime = true;
  static int p_cK = 0;
  
if ( GetTimeSampling() < 1e-8 ) { G4Exception( "GateVSourceVoxelReader::UpdateActivities", "UpdateActivities", FatalException, "Time Sampling too small.");return;}
 

if ( m_TimeActivTables.empty() == true) { //G4cout << " GateVSourceVoxelReader::UpdateActivities()  No time activity curves supplied." << G4endl;
                                         return;}

//ok we update for the current time all the activities for the translator

G4double currentTime = GateSourceMgr::GetInstance()->GetTime()/s ;

 cK = (G4int)( floor( currentTime / ( GetTimeSampling()/s)  ) ) + 1;

//G4cout << " GateVSourceVoxelReader::UpdateActivities  cK = " << cK<<"      p_cK = "<<p_cK<<G4endl;
//G4cout << " GateVSourceVoxelReader::UpdateActivities  IsFirstTime = " << IsFirstTime <<G4endl;

//G4cout << "GateVSourceVoxelReader::UpdateActivities(G4String, G4String) Time is " << currentTime<<"  cK = " << cK << "   p_cK = " << p_cK << G4endl;
//G4cout << " GetTimeSampling()/s " << GetTimeSampling()/s << G4endl;

if ( cK != p_cK ||  IsFirstTime == true  )
{

IsFirstTime = false;
G4cout << "GateVSourceVoxelReader::UpdateActivities(G4String, G4String) Time is " << currentTime<<"  cK = " << cK << "   p_cK = " << p_cK << G4endl;


if ( !m_TimeActivTables.empty() ) 
{

Initialize();

std::map< std::pair<G4double,G4double> , std::vector<std::pair<G4double,G4double> >  >::iterator iter;

std::vector<std::pair<G4double,G4double> > ActivCurve;

G4double Xd[400],Yd[400]; // data set points needed for interpolation
G4int npoints;

for ( iter = m_TimeActivTables.begin(); iter != m_TimeActivTables.end() ; iter++ )
 {

  ActivCurve.clear();

  ActivCurve = (*iter).second;

  npoints = ActivCurve.size();

  G4double activmin = ((*iter).first).first; 
  G4double activmax = ((*iter).first).second; 

  for ( G4int i = 0 ; i < npoints ; i++) // fill the points with the time curve activity values
    {
     Xd[i] = ActivCurve[i].first;
     Yd[i] = ActivCurve[i].second;
     
     G4cout << i << " x " << Xd[i]<<"  y "<<Yd[i]<<G4endl;
     
    }

  G4DataInterpolation AInterpolation(Xd, Yd, npoints , 0. , 0. ); // defines the interpolator

G4double current_activity  = AInterpolation.CubicSplineInterpolation( currentTime ); // interpolates

G4cout <<" current time "<< currentTime << " current activity " << current_activity<<G4endl;

  m_voxelTranslator->UpdateActivity( activmin, activmax , current_activity * becquerel ); // it associates to the translator key the new activity value

 }

G4cout << "   Description of Range Translator  " << G4endl;

//if (m_verboseLevel>1)
 m_voxelTranslator->Describe( 2 ) ;

ReadRTFile(HFN, FN);
//if (m_verboseLevel>1) 
Dump(0);
p_cK = cK;
}

}

}


void GateVSourceVoxelReader::UpdateActivities()
{

if ( m_TimeActivTables.empty() == true) { G4cout << " GateVSourceVoxelReader::UpdateActivities()  No time activity curves supplied." << G4endl;
                                          return;}

//ok we update for the current time all the activities for the translator

G4double currentTime = GateSourceMgr::GetInstance()->GetTime()/s ;


std::map< std::pair<G4double,G4double> , std::vector<std::pair<G4double,G4double> >  >::iterator iter;

std::vector<std::pair<G4double,G4double> > ActivCurve;

G4double Xd[400],Yd[400]; // data set points needed for interpolation
G4int npoints;

for ( iter = m_TimeActivTables.begin(); iter != m_TimeActivTables.end() ; iter++ )
 {

  ActivCurve.clear();

  ActivCurve = (*iter).second;

  npoints = ActivCurve.size();

  G4double  activmin = ((*iter).first).first; 
  G4double  activmax = ((*iter).first).second; 
  for ( G4int i = 0 ; i < npoints ; i++) // fill the points with the time curve activity values
    {
     Xd[i] = ActivCurve[i].first;
     Yd[i] = ActivCurve[i].second;
    }

  G4DataInterpolation AInterpolation(Xd, Yd, npoints , 0. , 0. ); // defines the interpolator

G4double current_activity  = AInterpolation.CubicSplineInterpolation( currentTime ); // interpolates

  m_voxelTranslator->UpdateActivity( activmin , activmax , current_activity * becquerel ); // it associates to the translator key the new activity value

 }

// G4cout << "   Description of Range Translator  " << G4endl;

m_voxelTranslator->Describe( 2 ) ;

}



void GateVSourceVoxelReader::SetTimeActivTables( G4String fileName)
{
 if ( m_voxelTranslator == 0 ) G4Exception("GateVSourceVoxelReader::SetTimeActivTables", "SetTimeActivTables", FatalException, " ERROR no translator found . Exiting." );
  m_TimeActivTables.clear();

  std::ifstream inFile;

  inFile.open(fileName.c_str(),std::ios::in);

if (!inFile.is_open())
  {G4cout << " Source Voxel Reader Message : ERROR - Time Activities Tables  file " << fileName << " not found " << G4endl;
   G4Exception( "GateVSourceVoxelReader::SetTimeActivTables", "SetTimeActivTables", FatalException, "Aborting...");
  }

  G4String fname;
  G4double activmin , activmax;
  G4int nTotCol;
  char buffer [200];

  inFile.getline(buffer,200);
  std::istringstream is(buffer);

  is >> nTotCol;

   G4cout << "==== Source Voxel Reader Time Activity Translation Table ====" << G4endl;
   G4cout << "number of couples to be read : " << nTotCol <<G4endl;

std::vector< std::pair<G4double,G4double> > m_ActivCurve;

for (G4int iCol=0; iCol<nTotCol; iCol++)
  {
    inFile.getline(buffer,200);
    is.clear();
    is.str(buffer);

    is >> activmin >> activmax >> fname;


       G4cout  << " activity range  [" << activmin <<" - " <<activmax << "] is associated  to Time Activity Curve read from file " << fname << G4endl;
      

    m_ActivCurve.clear();

// open file with fname file name which contains a time curve activity data set points

     std::ifstream TimeCurveFile;
     TimeCurveFile.open(fname.c_str(),std::ios::in);
     if (  !TimeCurveFile.is_open()  )
       {G4cout << " Source Voxel Reader Message : ERROR - Time Activities Tables  file " << fname << " not found " << G4endl;
        G4Exception( "GateVSourceVoxelReader::SetTimeActivTables", "SetTimeActivTables", FatalException, "Aborting...");
       }

   { ////// now we read from file fname the time activity data set points corresponding to the integer key value named activ
     ////
     //
     G4int nCol;
     char buf [200];
     G4double aTime, anActivity;
     TimeCurveFile.getline(buf,200);
     std::istringstream iss(buf);
     iss >> nCol;
     G4cout << "==== Source Voxel Reader Time Activity Curve Read From File "<< fname << "  ====" << G4endl;
     G4cout << "number of couples to be read : " << nCol <<G4endl;

     for (G4int iCol=0; iCol<nCol; iCol++)
      {
       TimeCurveFile.getline(buf,200);
       iss.clear();
       iss.str(buf);
       iss >> aTime >> anActivity;

       G4cout <<"At " << aTime << " seconds corresponds an Activity of " << anActivity << " Bcq" << G4endl;
      
       std::pair<G4double, G4double> couple( aTime, anActivity );
       m_ActivCurve.push_back( couple );
      }
   }

       TimeCurveFile.close();

    GateSourceVoxelRangeTranslator* theVT = dynamic_cast<GateSourceVoxelRangeTranslator*> ( m_voxelTranslator );
    if ( theVT != 0 ) theVT->AddTranslationRange(activmin, activmax ) ;
    std::pair<G4double, G4double> activrange( activmin, activmax );

    m_TimeActivTables.insert( make_pair( activrange , m_ActivCurve) );

  }

  G4cout << "========================================" << G4endl;

  inFile.close();

}
/* PY Descourt 08/09/2009 */
