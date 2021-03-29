/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateVSourceVoxelReader.hh"
#include "GateVSource.hh"
#include "GateSourceVoxelLinearTranslator.hh"
#include "GateSourceVoxelRangeTranslator.hh"
#include "GateSourceMgr.hh"
#include "GateImage.hh"

//-------------------------------------------------------------------------------------------------
GateVSourceVoxelReader::GateVSourceVoxelReader(GateVSource* source)
  : m_source(source)
  , m_voxelTranslator(0)
{
  m_position = G4ThreeVector();
  m_activityTotal = 0. * becquerel;
  m_tactivityTotal = 0. * becquerel;
//  m_activityMax   = 0. * becquerel;
  m_image_origin = G4ThreeVector(0);

  G4double voxelSize = 1.*mm;
  m_voxelSize = G4ThreeVector(voxelSize,voxelSize,voxelSize);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateVSourceVoxelReader::~GateVSourceVoxelReader()
{
  if (m_voxelTranslator) {
    delete m_voxelTranslator;
  }
  m_sourceVoxelIntegratedActivities.clear();
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSourceVoxelReader::Dump(G4int level)
{
  G4cout << "  Voxel reader ----------> " << m_type << Gateendl
         << "  number of voxels       : " << m_sourceVoxelActivities.size() << Gateendl
         << "  total activity (Bq)    : " << GetTotalActivity()/becquerel << Gateendl
         << "  position  (mm)         : "
         << GetPosition().x()/mm << " "
         << GetPosition().y()/mm << " "
         << GetPosition().z()/mm << Gateendl
         << "  voxel size  (mm)       : "
         << GetVoxelSize().x()/mm << " "
         << GetVoxelSize().y()/mm << " "
         << GetVoxelSize().z()/mm << Gateendl;

  if (level > 2) {
	  for (G4int iz=0; iz<m_voxelNz; iz++) {
		  for (G4int iy=0; iy<m_voxelNy; iy++) {
			  for (G4int ix=0; ix<m_voxelNx; ix++) {
				  G4cout << "   Index"
						  << " " << ix
						  << " " << iy
						  << " " << iz
						  << " Activity (Bq) " << m_sourceVoxelActivities[RealArrayIndex(ix,iy,iz)] / becquerel << Gateendl;
			  }
		  }
	  }
  }
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSourceVoxelReader::ExportSourceActivityImage(G4String activityImageFileName)
{
    if(!activityImageFileName.empty() && m_sourceVoxelActivities.size()!=0)
    {
        GateImage output;
        output.SetResolutionAndVoxelSize(G4ThreeVector(m_voxelNx,m_voxelNy,m_voxelNz),m_voxelSize);
        output.SetOrigin(m_image_origin);
        output.Allocate();

        GateImage::iterator po;
        po = output.begin();
        for(size_t i =0;i<m_sourceVoxelActivities.size();i++)
        {
            if(po != output.end()) {
                    *po = m_sourceVoxelActivities[i] / becquerel;
                    ++po;
            }
        }
        // Write image
        output.Write(activityImageFileName);
    }
}

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
G4int GateVSourceVoxelReader::GetNextSource()
{
  // the method decides which is the source that has to be used for this event
	G4int firstSource;

  if (m_sourceVoxelActivities.size()==0) {
    GateError("GateVSourceVoxelReader::GetNextSource : ERROR: No source available");
  } else {
    // if there is at least one voxel

    // now assign the event to one voxel, according to the relative activity
    // integral method
    // from STL doc: iterator upper_bound(const key_type& k)   Sorted Associative Container   Finds the first element whose key greater than k.
    firstSource = (m_sourceVoxelIntegratedActivities.upper_bound(G4UniformRand() * m_activityTotal))->second;

  }

  if (nVerboseLevel>1)
    G4cout << "GateVSourceVoxelReader::GetNextSource : source chosen : "
           << " " << GetVoxelIndices(firstSource)
           << Gateendl;

  return firstSource;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSourceVoxelReader::AddVoxel(G4int ix, G4int iy, G4int iz, G4double activity)
{ // no check if Voxel already existed to speed-up
  //
//  m_activityTotal += activity;
  m_sourceVoxelActivities[RealArrayIndex(ix,iy,iz)] = activity;


}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSourceVoxelReader::InsertTranslator(G4String translatorType)
{
  if (m_voxelTranslator) {
    GateError("GateVSourceVoxelReader::InsertTranslator: voxel translator already defined\n");
  } else {
    if (translatorType == G4String("linear")) {
      m_voxelTranslator = new GateSourceVoxelLinearTranslator(this);
    } else if (translatorType == G4String("range")) {
      m_voxelTranslator = new GateSourceVoxelRangeTranslator(this);
    } else {
      GateError("GateVSourceVoxelReader::InsertTranslator: unknown translator type\n");
    }
  }

}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSourceVoxelReader::RemoveTranslator()
{
  if (m_voxelTranslator) {
    delete m_voxelTranslator;
    m_voxelTranslator = 0;
  } else {
    GateError("GateVSourceVoxelReader::RemoveTranslator: voxel translator not defined\n");
  }
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSourceVoxelReader::PrepareIntegratedActivityMap()
{
  // erase all the elements of the old integrated activity map
  m_sourceVoxelIntegratedActivities.clear();

  // create the new integrated activity map
  m_activityTotal = 0.;
  //GateSourceActivityMap::iterator voxel;
  for (size_t iVoxel = 0; iVoxel < m_sourceVoxelActivities.size(); iVoxel++) {
	  if (m_sourceVoxelActivities[iVoxel]>0.0) {
		  m_activityTotal += m_sourceVoxelActivities[iVoxel];
		  m_sourceVoxelIntegratedActivities[m_activityTotal] = iVoxel;
	  }
  }

  if (nVerboseLevel>1) {
	  GateSourceIntegratedActivityMap::iterator intVoxel;
	  for (intVoxel = m_sourceVoxelIntegratedActivities.begin(); intVoxel != m_sourceVoxelIntegratedActivities.end(); intVoxel++) {
		  G4int iVoxel = intVoxel->second;
		  G4cout << "[GateVSourceVoxelReader::PrepareIntegratedActivityMap] "
				  << "   voxel: " << GetVoxelIndices(iVoxel)
				  << "   activity : (Bq) " << m_sourceVoxelActivities[iVoxel] / becquerel
				  << "   integrated: (Bq) " << ((*intVoxel).first)  / becquerel
				  << Gateendl;
    }
  }
  m_tactivityTotal = m_activityTotal;  // added by I. Martinez-Rovira (immamartinez@gmail.com)
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSourceVoxelReader::Initialize()
{
  m_sourceVoxelActivities.clear();
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
G4double GateVSourceVoxelReader::GetTimeSampling()
{
  return m_TS;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSourceVoxelReader::SetTimeSampling( G4double aTS )
{
  m_TS = aTS;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSourceVoxelReader::UpdateActivities(G4String  HFN, G4String FN )
{
  static G4bool IsFirstTime = true;
  static int p_cK = 0;

  if ( GetTimeSampling() < 1e-8 ) {
    GateError("GateVSourceVoxelReader::UpdateActivities : Time Sampling too small.");
    return;
  }

  if ( m_TimeActivTables.empty() == true) { return;}

  //ok we update for the current time all the activities for the translator

  G4double currentTime = GateSourceMgr::GetInstance()->GetTime()/s ;
  cK = (G4int)( floor( currentTime / ( GetTimeSampling()/s)  ) ) + 1;

  if ( cK != p_cK ||  IsFirstTime == true  )
    {
      IsFirstTime = false;
      G4cout << "GateVSourceVoxelReader::UpdateActivities(G4String, G4String) Time is " << currentTime<<"  cK = " << cK << "   p_cK = " << p_cK << Gateendl;

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
              ActivCurve = iter->second;
              npoints = ActivCurve.size();
              G4double activmin = (iter->first).first;
              G4double activmax = (iter->first).second;
              for ( G4int i = 0 ; i < npoints ; i++) // fill the points with the time curve activity values
                {
                  Xd[i] = ActivCurve[i].first;
                  Yd[i] = ActivCurve[i].second;
                  G4cout << i << " x " << Xd[i]<<"  y "<<Yd[i]<< Gateendl;
                }
              G4DataInterpolation AInterpolation(Xd, Yd, npoints , 0. , 0. ); // defines the interpolator
              G4double current_activity  = AInterpolation.CubicSplineInterpolation( currentTime ); // interpolates
              G4cout <<" current time "<< currentTime << " current activity " << current_activity<< Gateendl;

              m_voxelTranslator->UpdateActivity( activmin, activmax , current_activity * becquerel ); // it associates to the translator key the new activity value

            }

          G4cout << "   Description of Range Translator\n";

          //if (m_verboseLevel>1)
          m_voxelTranslator->Describe( 2 ) ;

          ReadRTFile(HFN, FN);
          //if (m_verboseLevel>1)
          Dump(0);
          p_cK = cK;
        }
    }
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSourceVoxelReader::UpdateActivities()
{
  if (m_TimeActivTables.empty() == true) {
    GateError(" GateVSourceVoxelReader::UpdateActivities()  No time activity curves supplied.\n");
    return;
  }

  //ok we update for the current time all the activities for the translator

  G4double currentTime = GateSourceMgr::GetInstance()->GetTime()/s ;
  std::map< std::pair<G4double,G4double> , std::vector<std::pair<G4double,G4double> >  >::iterator iter;
  std::vector<std::pair<G4double,G4double> > ActivCurve;

  G4double Xd[400],Yd[400]; // data set points needed for interpolation
  G4int npoints;

  for ( iter = m_TimeActivTables.begin(); iter != m_TimeActivTables.end() ; iter++ )
    {
      ActivCurve.clear();
      ActivCurve = iter->second;
      npoints = ActivCurve.size();
      G4double  activmin = (iter->first).first;
      G4double  activmax = (iter->first).second;
      for ( G4int i = 0 ; i < npoints ; i++) // fill the points with the time curve activity values
        {
          Xd[i] = ActivCurve[i].first;
          Yd[i] = ActivCurve[i].second;
        }
      G4DataInterpolation AInterpolation(Xd, Yd, npoints , 0. , 0. ); // defines the interpolator
      G4double current_activity  = AInterpolation.CubicSplineInterpolation( currentTime ); // interpolates
      m_voxelTranslator->UpdateActivity( activmin , activmax , current_activity * becquerel ); // it associates to the translator key the new activity value
    }
  // G4cout << "   Description of Range Translator  \n";
  m_voxelTranslator->Describe( 2 ) ;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSourceVoxelReader::SetTimeActivTables( G4String fileName)
{
  if ( m_voxelTranslator == 0 ) {
    GateError("GateVSourceVoxelReader::SetTimeActivTables : ERROR no translator found . Exiting.");
  }
  m_TimeActivTables.clear();

  std::ifstream inFile;
  inFile.open(fileName.c_str(),std::ios::in);

  if (!inFile.is_open())
    {G4cout << " Source Voxel Reader Message : ERROR - Time Activities Tables  file " << fileName << " not found \n";
      G4Exception( "GateVSourceVoxelReader::SetTimeActivTables", "SetTimeActivTables", FatalException, "Aborting...");
    }

  G4String fname;
  G4double activmin , activmax;
  G4int nTotCol;
  char buffer [200];

  inFile.getline(buffer,200);
  std::istringstream is(buffer);

  is >> nTotCol;

  G4cout << "==== Source Voxel Reader Time Activity Translation Table ====\n";
  G4cout << "number of couples to be read : " << nTotCol << Gateendl;

  std::vector< std::pair<G4double,G4double> > m_ActivCurve;

  for (G4int iCol=0; iCol<nTotCol; iCol++)
    {
      inFile.getline(buffer,200);
      is.clear();
      is.str(buffer);

      is >> activmin >> activmax >> fname;


      G4cout  << " activity range  [" << activmin <<" - " <<activmax << "] is associated  to Time Activity Curve read from file " << fname << Gateendl;


      m_ActivCurve.clear();

      // open file with fname file name which contains a time curve activity data set points

      std::ifstream TimeCurveFile;
      TimeCurveFile.open(fname.c_str(),std::ios::in);
      if (  !TimeCurveFile.is_open()  )
        {G4cout << " Source Voxel Reader Message : ERROR - Time Activities Tables  file " << fname << " not found \n";
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
        G4cout << "==== Source Voxel Reader Time Activity Curve Read From File "<< fname << "  ====\n";
        G4cout << "number of couples to be read : " << nCol << Gateendl;

        for (G4int iCol=0; iCol<nCol; iCol++)
          {
            TimeCurveFile.getline(buf,200);
            iss.clear();
            iss.str(buf);
            iss >> aTime >> anActivity;

            G4cout <<"At " << aTime << " seconds corresponds an Activity of " << anActivity << " Bcq\n";

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

  G4cout << "========================================\n";

  inFile.close();

}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
G4ThreeVector GateVSourceVoxelReader::ComputeSourcePositionFromIsoCenter(G4ThreeVector p)
{
  G4ThreeVector t = m_image_origin - p;
  return t;
}
//-------------------------------------------------------------------------------------------------
