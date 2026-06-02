/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details

Contact: Olga Kochebina, kochebina@gmail.com
or
Mohammad Mirdoraghi, mirdoraghimohammad@gmail.com
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
GateParameterisedPinholeCollimator::~GateParameterisedPinholeCollimator()
{
  delete m_messenger;
}

//-------------------------------------------------------------------------------------------------------------------
void GateParameterisedPinholeCollimator::ResizeCollimator()
{
  // Implementation as needed for dynamic resizing
}

//-------------------------------------------------------------------------------------------------------------------
G4LogicalVolume* GateParameterisedPinholeCollimator::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  G4ThreeVector BoxPos_down;
  G4ThreeVector BoxPos_up;

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
	
	std::ifstream fin;
	fin.open(m_InputFile);

	fin >> temp_str;
	if (temp_str[0] == '#')
		fin.ignore(256,'\n'); 

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

	G4SubtractionSolid* temp_solid;
	temp_solid=(G4SubtractionSolid*) m_colli_solid;

	float z =  GetCollimatorRotRadius();
	float Dz;
	G4ThreeVector normal;
	float h = GetCollimatorHeight()/2.;
	
	for(int i=0;i<n_pinholes;i++)
	{
	  fin >> x >> y >> dia  >> cone_angle >> x_focal >> y_focal;
	  
	  // 1. Correct half-angle alpha in radians
	  float alpha = (cone_angle / 2.0f) * (pi / 180.0f);
	  
	  // 2. TRUE 3D Tilt Angle (beta) 
	  // This fixes the error where beta only considered the X-axis.
	  float dx = x - x_focal;
	  float dy = y - y_focal;
	  float r = sqrt(dx*dx + dy*dy);
	  float beta = (r == 0) ? (pi/2.0f) : atan(z / r);

	  // 3. Geometric parameters along the tilted axis
	  float a, b, l, k, s, n;
	  float tan_plus = tan(beta + alpha);
	  float tan_minus = tan(beta - alpha);

	  k = dia * tan_minus * tan_plus / (tan_plus - tan_minus);
	  n = (k / tan_plus) - dia;
	  a = k / sin(beta);
	  l = h / sin(beta);
	  s = (k / tan(beta)) - n;
	  b = (dia - s) * (a + l) * cos(beta) / a;

	  // 4. FINAL DZ CALCULATION (Corrected scalar depth)
	  Dz = a + l + b;

	  // 5. Create Unique Names for Visualization/Debugging
	  G4String upName = GetSolidName() + "_up_" + std::to_string(i);
	  G4String downName = GetSolidName() + "_down_" + std::to_string(i);

	  // 6. Safety check for rmax to prevent degenerate volumes
	  G4double rmax = Dz * tan(alpha);  
          if (rmax < (dia/2.0)) rmax = (dia/2.0) + 0.01*mm;

	  m_cone_up_solid
	    = new G4Cons(upName, 0, rmax*mm, 0, 0, (Dz/2.)*mm, 0.*deg, 360.*deg); // Toward detector
	  
	  m_cone_down_solid
	    = new G4Cons(downName, 0, 0, 0, rmax*mm, (Dz/2.)*mm, 0.*deg, 360.*deg); // Toward source

	  // 7. Persistent Rotation Matrix
	  // Heap allocation prevents all pinholes from sharing a single local matrix reference.
	  G4RotationMatrix* pRot = new G4RotationMatrix();
	  normal.set(dx, dy, z);
	  normal /= normal.mag();

	  pRot->rotateY(atan2(normal.x(), normal.z())*rad);
	  pRot->rotateX(-atan2(normal.y(), normal.z())*rad);

	  // 8. Balanced 3D Displacement
	  // Ensures the waist remains anchored at (x, y) on the midplane.
	  float correction_z = k - (Dz/2.0f) * sin(beta);
          float horizontal_shift = (beta == pi/2.0f) ? 0 : correction_z / tan(beta);
          float phi_dir = atan2(dy, dx); 

	  BoxPos_down.set(x + horizontal_shift*cos(phi_dir), y + horizontal_shift*sin(phi_dir), -correction_z);
	  BoxPos_up.set(x - horizontal_shift*cos(phi_dir), y - horizontal_shift*sin(phi_dir), correction_z);

	  // 9. Boolean Subtraction
	  m_sub_up_solid
	    = new G4SubtractionSolid(upName + "_sub", temp_solid, m_cone_up_solid, pRot, BoxPos_up);
	  
	  m_sub_down_solid
	     = new G4SubtractionSolid(downName + "_sub", m_sub_up_solid, m_cone_down_solid, pRot, BoxPos_down);
	
	  temp_solid = m_sub_down_solid;
	}

	fin.close();
	m_colli_log = new G4LogicalVolume(temp_solid, mater, GetLogicalVolumeName(), 0, 0, 0);
    }

  return m_colli_log;
}

void GateParameterisedPinholeCollimator::DestroyOwnSolidAndLogicalVolume()
{
  if (m_colli_log)
    delete m_colli_log;
  m_colli_log = 0;
}
