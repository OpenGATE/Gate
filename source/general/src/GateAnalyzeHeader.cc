/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateAnalyzeHeader.hh"
#include "GateMachine.hh"

/*
/// Acceptable values for hdr.ic.datatype
#define HDRH_DT_UNKNOWN			0
#define HDRH_DT_BINARY			1
#define HDRH_DT_UNSIGNED_CHAR	2
#define HDRH_DT_SIGNED_SHORT	4
#define HDRH_DT_SIGNED_INT		8
#define HDRH_DT_FLOAT			16
#define HDRH_DT_COMPLEX			32
#define HDRH_DT_DOUBLE			64
#define HDRH_DT_RGB				128
#define HDRH_DT_ALL				255
*/

//-----------------------------------------------------------------------------
const GateAnalyzeHeader::TypeCode GateAnalyzeHeader::UnknownType = 0;
const GateAnalyzeHeader::TypeCode GateAnalyzeHeader::BinaryType = 1;
const GateAnalyzeHeader::TypeCode GateAnalyzeHeader::UnsignedCharType = 2;
const GateAnalyzeHeader::TypeCode GateAnalyzeHeader::SignedShortType = 4;
const GateAnalyzeHeader::TypeCode GateAnalyzeHeader::SignedIntType = 8;
const GateAnalyzeHeader::TypeCode GateAnalyzeHeader::FloatType = 16;
const GateAnalyzeHeader::TypeCode GateAnalyzeHeader::ComplexType = 32;
const GateAnalyzeHeader::TypeCode GateAnalyzeHeader::DoubleType = 64;
const GateAnalyzeHeader::TypeCode GateAnalyzeHeader::RGBType = 128;
const GateAnalyzeHeader::TypeCode GateAnalyzeHeader::AllType = 255;
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateAnalyzeHeader::GateAnalyzeHeader()
{
  SetDefaults();
}
//----------------------------------------------------------------------------- 

//-----------------------------------------------------------------------------
void GateAnalyzeHeader::SetDefaults()
{
  m_data.hk.sizeof_hdr = 348;
  //for(int i = 0;i<10;i++) m_data.hk.data_type[i]=;
  strcpy(m_data.hk.data_type,"dsr");
  strcpy(m_data.hk.db_name,"/home");
  m_data.hk.extents = 0;
  m_data.hk.session_error = 0;
  m_data.hk.regular = 'r';
  m_data.hk.hkey_un0 = '0';
	
  /* image dimension */
  m_data.ic.dim[0] = 4;
  m_data.ic.dim[1] = 0; 
  m_data.ic.dim[2] = 0;  
  m_data.ic.dim[3] = 0; 
  m_data.ic.dim[4] = 1;
  m_data.ic.dim[5] = 0;
  m_data.ic.dim[6] = 0;
  m_data.ic.dim[7] = 0;
	
  strcpy(m_data.ic.vox_units,"mm");
  strcpy(m_data.ic.cal_units,"Bq/cc");
  // I USE UNUSED1 TO STORE THE NUMBER OF CHANNELS
  m_data.ic.unused1 = 1;
  m_data.ic.datatype = UnknownType;
  m_data.ic.bitpix = 0;
  m_data.ic.dim_un0 = 0;

  m_data.ic.pixdim[0] =  0.0;
  m_data.ic.pixdim[1] =  1.0;
  m_data.ic.pixdim[2] =  1.0;
  m_data.ic.pixdim[3] =  1.0;
  m_data.ic.pixdim[4] =  0.0;
  m_data.ic.pixdim[5] =  0.0;
  m_data.ic.pixdim[6] =  0.0;
  m_data.ic.pixdim[7] =  0.0;
  m_data.ic.vox_offset = 0.0;
  m_data.ic.funused1 = 1.0;
  m_data.ic.funused2 = 0.0;
  m_data.ic.funused3 = 0.0;
  m_data.ic.cal_max = 0.0;
  m_data.ic.cal_min = 0.0;
  m_data.ic.compressed = 0;
  m_data.ic.verified = 0;
  m_data.ic.glmax = 32767;
  m_data.ic.glmin =0;

  /* data_history */
  strcpy(m_data.hist.descrip,"image");
  strcpy(m_data.hist.aux_file,"none");
  m_data.hist.orient='0';                
  strcpy(m_data.hist.originator,"none");  
  strcpy(m_data.hist.generated,"none"); 
  strcpy(m_data.hist.scannum,"none");   
  strcpy(m_data.hist.patient_id,"none");    
  strcpy(m_data.hist.exp_date,"none");  
  strcpy(m_data.hist.exp_time,"none");  
  strcpy(m_data.hist.hist_un0,"no");  
  m_data.hist.views = 0;       
  m_data.hist.vols_added = 0;  
  m_data.hist.start_field = 0;  
  m_data.hist.field_skip = 0;   
  m_data.hist.omax = 0;
  m_data.hist.omin = 0;    
  m_data.hist.smax = 0;
  m_data.hist.smin = 0; 

  // Initialize
//   strcpy(m_data.hist.orient, "0");
//   m_data.hist.originator
// m_data.hist.generated
// m_data.hist.scannum

}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
bool GateAnalyzeHeader::Read( const G4String& name)
{
  FILE *fph=0;
  if ( (fph=fopen(name,"rb")) == NULL) return false;

  /// Tests wether the bytes are written in reverse order than the current machine settings
	/// (little/big endians)
	/// The first record should be the file size, hence if reading it doesn't match 
	/// the actual file size assume the header has bytes inverted
	/// gets the file size
	fseek(fph,0, SEEK_END);			
	int file_size = ftell(fph);   
	fseek(fph,0, SEEK_SET);		
	
	/// reads the records
if(	  fread(&m_data.hk.sizeof_hdr, sizeof(int), 1, fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.hk.data_type,sizeof(char),10,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.hk.db_name,sizeof(char),18,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.hk.extents,sizeof(int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.hk.session_error,sizeof(short int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.hk.regular,sizeof(char),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.hk.hkey_un0,sizeof(char),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.ic.dim,sizeof(short int),8,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.ic.vox_units,sizeof(char),4,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.ic.cal_units,sizeof(char),8,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.ic.unused1,sizeof(short int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.ic.datatype,sizeof(short int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.ic.bitpix,sizeof(short int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.ic.dim_un0,sizeof(short int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.ic.pixdim,sizeof(float),8,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.ic.vox_offset,sizeof(float),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.ic.funused1,sizeof(float),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.ic.funused2,sizeof(float),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.ic.funused3,sizeof(float),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.ic.cal_max,sizeof(float),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.ic.cal_min,sizeof(float),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.ic.compressed,sizeof(int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.ic.verified,sizeof(int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.ic.glmax,sizeof(int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.ic.glmin,sizeof(int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.hist.descrip,sizeof(char),80,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.hist.aux_file,sizeof(char),24,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.hist.orient,sizeof(char),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.hist.originator,sizeof(char),10,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.hist.generated,sizeof(char),10,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.hist.scannum,sizeof(char),10,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.hist.patient_id,sizeof(char),10,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.hist.exp_date,sizeof(char),10,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.hist.exp_time,sizeof(char),10,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(m_data.hist.hist_un0,sizeof(char),3,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.hist.views,sizeof(int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.hist.vols_added,sizeof(int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.hist.start_field,sizeof(int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.hist.field_skip,sizeof(int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.hist.omax,sizeof(int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.hist.omin,sizeof(int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.hist.smax,sizeof(int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
if(	  fread(&m_data.hist.smin,sizeof(int),1,fph) == 0 ){G4cerr << "Problem reading data!!!" << G4endl;}
	  fclose(fph),fph=NULL;


	  //std::cout << "m_data.hk.data_type=" <<m_data.hk.data_type << std::endl; 

	  m_rightEndian = TRUE;
	  /// if opposite endian : reverse all data
		// std::cout << "m_data.hk.sizeof_hdr =" << m_data.hk.sizeof_hdr
		// 			  << " file_size=" << file_size << std::endl;

		if( m_data.hk.sizeof_hdr != file_size) {
		  m_rightEndian = FALSE;
		  /// swaps the records
			GateMachine::SwapEndians(&m_data.hk.sizeof_hdr, 1);
			GateMachine::SwapEndians(m_data.hk.data_type,10);
			GateMachine::SwapEndians(m_data.hk.db_name,18);
			GateMachine::SwapEndians(&m_data.hk.extents,1);
			GateMachine::SwapEndians(&m_data.hk.session_error,1);
			GateMachine::SwapEndians(&m_data.hk.regular,1);
			GateMachine::SwapEndians(&m_data.hk.hkey_un0,1);
			GateMachine::SwapEndians(m_data.ic.dim,8);
			GateMachine::SwapEndians(m_data.ic.vox_units,4);
			GateMachine::SwapEndians(m_data.ic.cal_units,8);
			GateMachine::SwapEndians(&m_data.ic.unused1,1);
			GateMachine::SwapEndians(&m_data.ic.datatype,1);
			GateMachine::SwapEndians(&m_data.ic.bitpix,1);
			GateMachine::SwapEndians(&m_data.ic.dim_un0,1);
			GateMachine::SwapEndians(m_data.ic.pixdim,8);
			GateMachine::SwapEndians(&m_data.ic.vox_offset,1);
			GateMachine::SwapEndians(&m_data.ic.funused1,1);
			GateMachine::SwapEndians(&m_data.ic.funused2,1);
			GateMachine::SwapEndians(&m_data.ic.funused3,1);
			GateMachine::SwapEndians(&m_data.ic.cal_max,1);
			GateMachine::SwapEndians(&m_data.ic.cal_min,1);
			GateMachine::SwapEndians(&m_data.ic.compressed,1);
			GateMachine::SwapEndians(&m_data.ic.verified,1);
			GateMachine::SwapEndians(&m_data.ic.glmax,1);
			GateMachine::SwapEndians(&m_data.ic.glmin,1);
			GateMachine::SwapEndians(m_data.hist.descrip,80);
			GateMachine::SwapEndians(m_data.hist.aux_file,24);
			GateMachine::SwapEndians(&m_data.hist.orient,1);
			GateMachine::SwapEndians(m_data.hist.originator,10);
			GateMachine::SwapEndians(m_data.hist.generated,10);
			GateMachine::SwapEndians(m_data.hist.scannum,10);
			GateMachine::SwapEndians(m_data.hist.patient_id,10);
			GateMachine::SwapEndians(m_data.hist.exp_date,10);
			GateMachine::SwapEndians(m_data.hist.exp_time,10);
			GateMachine::SwapEndians(m_data.hist.hist_un0,3);
			GateMachine::SwapEndians(&m_data.hist.views,1);
			GateMachine::SwapEndians(&m_data.hist.vols_added,1);
			GateMachine::SwapEndians(&m_data.hist.start_field,1);
			GateMachine::SwapEndians(&m_data.hist.field_skip,1);
			GateMachine::SwapEndians(&m_data.hist.omax,1);
			GateMachine::SwapEndians(&m_data.hist.omin,1);
			GateMachine::SwapEndians(&m_data.hist.smax,1);
			GateMachine::SwapEndians(&m_data.hist.smin,1);	
		}
		///	
		  return true;
	
}
//-----------------------------------------------------------------------------


/*
//-----------------------------------------------------------------------------
/// Sets the header information to the image's characteristics
void GateAnalyzeHeader::setData( const Image& i )
{
// I USE UNUSED1 TO STORE THE NUMBER OF CHANNELS
m_data.ic.unused1 = i.size()(0);
m_data.ic.dim[1] = i.size()(1);
m_data.ic.dim[2] = i.size()(2);
m_data.ic.dim[3] = i.size()(3);
m_data.ic.datatype = basicTypeCodeToAnalyzeCode(i.type());
m_data.ic.bitpix = BasicType::size(i.type());

//	lglLOG("basicTypeCode = "<<i.type()<<ENDL);
//	lglLOG("AnalyzeCode   = "<<m_data.ic.datatype<<ENDL);
//	lglLOG("Analyze hdr write : bitpix="<<m_data.ic.bitpix<<ENDL);
}
//-----------------------------------------------------------------------------
*/

//-----------------------------------------------------------------------------
/// Writes the header 
bool GateAnalyzeHeader::Write( const G4String& filename )
{
  FILE *fph=0;
  if ( (fph=fopen(filename,"wb")) == NULL) {
    return false;
  }
  fwrite(&m_data.hk.sizeof_hdr,sizeof(int),1,fph);
  fwrite(m_data.hk.data_type,sizeof(char),10,fph);
  fwrite(m_data.hk.db_name,sizeof(char),18,fph);
  fwrite(&m_data.hk.extents,sizeof(int),1,fph);
  fwrite(&m_data.hk.session_error,sizeof(short int),1,fph);
  fwrite(&m_data.hk.regular,sizeof(char),1,fph);
  fwrite(&m_data.hk.hkey_un0,sizeof(char),1,fph);
	
  fwrite(m_data.ic.dim,sizeof(short int),8,fph);
  fwrite(m_data.ic.vox_units,sizeof(char),4,fph);
  fwrite(m_data.ic.cal_units,sizeof(char),8,fph);
  fwrite(&m_data.ic.unused1,sizeof(short int),1,fph);
  fwrite(&m_data.ic.datatype,sizeof(short int),1,fph);
  fwrite(&m_data.ic.bitpix,sizeof(short int),1,fph);
  fwrite(&m_data.ic.dim_un0,sizeof(short int),1,fph);
  fwrite(m_data.ic.pixdim,sizeof(float),8,fph);
  fwrite(&m_data.ic.vox_offset,sizeof(float),1,fph);
  fwrite(&m_data.ic.funused1,sizeof(float),1,fph);
  fwrite(&m_data.ic.funused2,sizeof(float),1,fph);
  fwrite(&m_data.ic.funused3,sizeof(float),1,fph);
  fwrite(&m_data.ic.cal_max,sizeof(float),1,fph);
  fwrite(&m_data.ic.cal_min,sizeof(float),1,fph);
  fwrite(&m_data.ic.compressed,sizeof(int),1,fph);
  fwrite(&m_data.ic.verified,sizeof(int),1,fph);
  fwrite(&m_data.ic.glmax,sizeof(int),1,fph);
  fwrite(&m_data.ic.glmin,sizeof(int),1,fph);
	
  fwrite(m_data.hist.descrip,sizeof(char),80,fph);
  fwrite(m_data.hist.aux_file,sizeof(char),24,fph);
  fwrite(&m_data.hist.orient,sizeof(char),1,fph);
  fwrite(m_data.hist.originator,sizeof(char),10,fph);
  fwrite(m_data.hist.generated,sizeof(char),10,fph);
  fwrite(m_data.hist.scannum,sizeof(char),10,fph);
  fwrite(m_data.hist.patient_id,sizeof(char),10,fph);
  fwrite(m_data.hist.exp_date,sizeof(char),10,fph);
  fwrite(m_data.hist.exp_time,sizeof(char),10,fph);
  fwrite(m_data.hist.hist_un0,sizeof(char),3,fph);
  fwrite(&m_data.hist.views,sizeof(int),1,fph);
  fwrite(&m_data.hist.vols_added,sizeof(int),1,fph);
  fwrite(&m_data.hist.start_field,sizeof(int),1,fph);
  fwrite(&m_data.hist.field_skip,sizeof(int),1,fph);
  fwrite(&m_data.hist.omax,sizeof(int),1,fph);
  fwrite(&m_data.hist.omin,sizeof(int),1,fph);
  fwrite(&m_data.hist.smax,sizeof(int),1,fph);
  fwrite(&m_data.hist.smin,sizeof(int),1,fph);

  fclose(fph), fph=NULL;
  return true;		
}
//-----------------------------------------------------------------------------





// EOF

