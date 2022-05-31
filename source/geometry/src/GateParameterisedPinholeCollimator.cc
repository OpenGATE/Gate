/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details

Contact: Olga Kochebina, kochebina@gmail.com

----------------------*/


#include "GateParameterisedPinholeCollimator.hh"
#include "GateParameterisedPinholeCollimatorMessenger.hh"

#include "GateTrpd.hh"
#include "G4Cons.hh"
#include "GateParameterisedHole.hh"
#include "GateObjectChildList.hh"
#include "GateMaterialDatabase.hh"

#include "G4UnitsTable.hh"
#include "G4VisAttributes.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"

//-------------------------------------------------------------------------------------------------------------------
GateParameterisedPinholeCollimator::GateParameterisedPinholeCollimator(const G4String& itsName,
							 G4bool acceptsChildren, 
		 			 		 G4int depth)
  : GateTrpd(itsName,"Air",8.0,8.4,1.,acceptsChildren,depth),
    m_colli_solid(0), m_messenger(0)
   
{ 
  G4cout << " Constructor GateParameterisedPinholeCollimator - begin " << itsName << Gateendl;
  
  m_InputFile = "";
  m_Height=1.*cm;
  m_RotRadius=1.*cm;
  m_DimensionX1=1.*cm;
  m_DimensionY1=1.*cm;
  m_DimensionX2=1.*cm;
  m_DimensionY2=1.*cm;

  
  m_messenger = new GateParameterisedPinholeCollimatorMessenger(this);
  
  G4cout << " Constructor GateParameterisedPinholeCollimator - end \n";
}
//-------------------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------
GateParameterisedPinholeCollimator::GateParameterisedPinholeCollimator(const G4String& itsName,const G4String& itsMaterialName,
								       const G4String& itsInputFile, 
								       G4double itsHeight,G4double itsRotRadius,
								       G4double itsDimensionX1,G4double itsDimensionY1,
								       G4double itsDimensionX2,G4double itsDimensionY2, G4bool itsTest, G4double itsPinholeDia)
  : GateTrpd(itsName,itsMaterialName,itsDimensionX1,itsDimensionY1,itsDimensionX2,itsDimensionY2,itsHeight,false,false),
    m_colli_solid(0), m_colli_log(0), m_messenger(0)
{

  
  m_InputFile = "";
  m_Height=itsHeight;
  m_RotRadius=itsRotRadius;
  m_DimensionX1=itsDimensionX1;
  m_DimensionY1=itsDimensionY1;
  m_DimensionX2=itsDimensionX2;
  m_DimensionY2=itsDimensionY2;

  m_messenger = new GateParameterisedPinholeCollimatorMessenger(this);
}
//-------------------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------
GateParameterisedPinholeCollimator::~GateParameterisedPinholeCollimator()
{
  delete m_messenger;
}
//-------------------------------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------------------------------
void GateParameterisedPinholeCollimator::ResizeCollimator()
{
  //GetCollimatorCreator()->SetBoxXLength(m_DimensionX1);
  //GetCollimatorCreator()->SetBoxYLength(m_DimensionY1);
  //GetCollimatorCreator()->SetBoxZLength(m_Height);

  // m_holeInserter->ResizeHole(m_FocalDistanceX,m_FocalDistanceY,m_SeptalThickness,m_InnerRadius,
  //		     m_Height,m_DimensionX1,m_DimensionY1);
}
//-------------------------------------------------------------------------------------------------------------------


G4LogicalVolume* GateParameterisedPinholeCollimator::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{

  G4ThreeVector BoxPos_down;
  G4ThreeVector BoxPos_up;
  G4RotationMatrix rotMatrix; // Unit matrix created by default

  if (!flagUpdateOnly)
    {
      m_colli_solid
	= new G4Trd(GetSolidName(), GetCollimatorDimensionX1()/2., GetCollimatorDimensionX2()/2., GetCollimatorDimensionY1()/2., GetCollimatorDimensionY2()/2.,
		    GetCollimatorHeight()/2.);

      
	char temp_str[64];

	int n_pinholes;
	float x, y;
	float dia, cone_angle;
	float x_focal, y_focal;
	
      // open pinhole description file
	std::ifstream fin;
	fin.open(m_InputFile);

	// parse file
	fin >> temp_str;
	if (temp_str[0] == '#')
		fin.ignore(256,'\n'); // ignore first line if it starts with #

	fin >> temp_str;
	if(temp_str[0] == '[')
		std::cout << "\nLoading pinhole definition " << temp_str << std::endl;
	else
	{
		std::cout << "\nInvalid pinhole definition file." << std::endl;
		fin.close();
		return NULL;
	}

	fin >> n_pinholes;

	//Temp. volume of pinholes
	G4SubtractionSolid* temp_solid;
	temp_solid=(G4SubtractionSolid*) m_colli_solid;


	
	float z =  GetCollimatorRotRadius();
	float Dz;
	G4ThreeVector normal;
	
	float h = GetCollimatorHeight()/2.;
	
	for(int i=0;i<n_pinholes;i++)
	{
	  
	  fin >> x >> y >> dia  >> cone_angle >> x_focal >> y_focal;
	  cone_angle *= 4.0f * atan(1.0f) / 180.0f;
	  
	  float alpha=cone_angle;
	  normal.setX(x-x_focal);
	  normal.setY(y-y_focal);
	  normal.setZ(z);
	  

	  float t=normal.getX()*normal.getX()+normal.getY()*normal.getY()+normal.getZ()*normal.getZ();
	  normal.setX(normal.getX()/sqrt(t));
	  normal.setY(normal.getY()/sqrt(t));
	  normal.setZ(normal.getZ()/sqrt(t));
	 
	  float theta = atan((x-x_focal)/z);  
	  float beta= fabs(pi/2. - fabs(theta));

	  float a,b,l,k,s,n;
	  
	  float tan_plus = tan(beta+alpha);
	  float tan_minus = tan(beta-alpha);

	  k = dia * tan_minus * tan_plus / (tan_plus-tan_minus) ;
	  n = k/tan_minus - dia;

	
	  a = k/sin(beta);
	  l = h/sin(beta);
	  s = k/tan(beta) - n ;
	  b = (dia-s) * (a+l) * cos(beta) /(a); 
	  // G4cout<<i<<" "<< x<<" "<<y <<"; "<< b <<" "<<  (dia-s) * (a+l) /(a*tan(beta))<<G4endl;


	  float Dz_x = a + l + b;
	  float Dz_y = (y-y_focal) / cos((y-y_focal)/z);
	  Dz = sqrt(Dz_x*Dz_x + Dz_y*Dz_y);



	  G4double rmax=Dz*tanf(alpha);  
	
	  m_cone_up_solid
	    = new G4Cons(GetSolidName(),0,rmax*mm,0,0,(Dz/2.)*mm, 0.*deg, 360.*deg);// toward the detector		    
	  
	  m_cone_down_solid
	    = new G4Cons(GetSolidName(),0,0,0,rmax*mm, (Dz/2.)*mm, 0.*deg, 360.*deg);// toward the source
	 
	  float correction_z=k-(Dz/2.)*sin(beta);
	  float correction_x=correction_z*(x-x_focal)/z;
	  float correction_y=correction_z*(y-y_focal)/z;

	  BoxPos_down.setX(x+correction_x);
	  BoxPos_down.setY(y+correction_y);
	  BoxPos_down.setZ(-correction_z);
	 
	  BoxPos_up.setX(x-correction_x);
	  BoxPos_up.setY(y-correction_y);
	  BoxPos_up.setZ(correction_z);
	  
	  rotMatrix.rotateY(atan(normal.getX()/normal.getZ())*rad);
	  rotMatrix.rotateX(-atan(normal.getY()/normal.getZ())*rad);
	  
	  
	  //G4cout<<i<<" "<< x<<" "<<y <<"; "<< b <<" "<< rotMatrix.phi() *180.0f/(4.0f * atan(1.0f))<<" "<< rotMatrix.theta()*180.0f/(4.0f * atan(1.0f))<<" "<<rotMatrix.psi()*180.0f/(4.0f * atan(1.0f))<<G4endl;
	   	
	  m_sub_up_solid
	    = new G4SubtractionSolid(GetSolidName(), temp_solid, m_cone_up_solid, &rotMatrix, BoxPos_up);
	  
	  m_sub_down_solid
	     = new G4SubtractionSolid(GetSolidName(), m_sub_up_solid, m_cone_down_solid, &rotMatrix, BoxPos_down);
	  
	  G4ThreeVector u = G4ThreeVector(1, 0, 0);
	  G4ThreeVector v = G4ThreeVector(0, 1, 0);
	  G4ThreeVector w = G4ThreeVector(0, 0, 1);
	  rotMatrix.setRows(u, v, w);
	  
	
	  temp_solid=m_sub_down_solid;
	}
	//	exit(1);

	fin.close();

	//m_colli_pinholes_log
	m_colli_log
	  = new G4LogicalVolume( temp_solid, mater, GetLogicalVolumeName(),0,0,0);

    }

  //return m_colli_pinholes_log;
  return m_colli_log;
}

void GateParameterisedPinholeCollimator::DestroyOwnSolidAndLogicalVolume()
{

  if (m_colli_log)
    delete m_colli_log;
  m_colli_log = 0;


}
