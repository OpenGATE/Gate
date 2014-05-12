/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! 
  \brief Handles the basic machine properties (e.g. endianness) and machine-dependent operations. 
  \brief By laurent.guigues@creatis.insa-lyon.fr
*/

#ifndef __GateMachine_h__
#define __GateMachine_h__

/// \brief Handles the basic machine properties (e.g. endianness, bus size) and machine-dependent operations. 
class GateMachine
{
public:
  /// Type of machine */
  //@{
  /// Determines the type of Endians (little=Intel like / big=Sun like)
  inline static void TestEndians();
  /// Is the machine an Intel like processor ? (equivalent to IsLittleEndians)
  inline static bool IsIntel () { return m_Intel ; }
  /// Is the machine an Intel like processor ?
  inline static bool IsLittleEndians () { return m_Intel ; }
  /// Is the machine a Sun like processor ? (== IsBigEndians)
  inline static bool IsSun () { return !m_Intel ; }
  /// Is the machine a Sun like processor ? 
  inline static bool IsBigEndians () { return !m_Intel ; }
  ///
  inline static void ChangeEndians() { m_Intel = ! m_Intel; }
  inline static void SimulateIntel() { m_Intel = true; }
  inline static void SimulateSun() { m_Intel = true; }

  /// Test the size of the bus
  inline static void TestBusSize();
  /// Returns true iff the machine has a 32 bits processor
  inline static bool Is32bits () { return m_32 ; }
  /// Returns true iff the machine has a 64 bits processor
  inline static bool Is64bits () { return m_64 ; }
  //@}

  ///@Little/big endians swap methods 
  ///@{
  ///
  inline static void SwapEndians (bool &) {}
  ///
  inline static void SwapEndians (char &) {}
  ///
  inline static void SwapEndians (signed char &) {}
  ///
  inline static void SwapEndians (unsigned char &) {}
  ///
  inline static void SwapEndians (signed short &) ;
  ///
  inline static void SwapEndians (unsigned short &) ;
  ///
  inline static void SwapEndians (signed int &) ;
  ///
  inline static void SwapEndians (unsigned int &) ;
  ///
  inline static void SwapEndians (signed long &) ;
  ///
  inline static void SwapEndians (unsigned long & ) ;
  ///
  inline static void SwapEndians (float &) ;
  ///
  inline static void SwapEndians (double & ) ;
  ///
  inline static void SwapEndians (bool *, long) {}
  ///
  inline static void SwapEndians (char *, long) {}
  ///
  inline static void SwapEndians (signed char *, long) {}
  ///
  inline static void SwapEndians (unsigned char *, long) {}
  ///
  inline static void SwapEndians (signed short *, long) ;
  ///
  inline static void SwapEndians (unsigned short *, long) ;
  ///
  inline static void SwapEndians (signed int *, long) ;
  ///
  inline static void SwapEndians (unsigned int *, long) ;
  ///
  inline static void SwapEndians (signed long *, long) ;
  ///
  inline static void SwapEndians (unsigned long *, long) ;
  ///
  inline static void SwapEndians (float *, long) ;
  ///
  inline static void SwapEndians (double *, long) ;

  ///@}
protected:
  static bool m_Intel;
  static bool m_32;
  static bool m_64;
};
//-----------------------------------------------------------------------------
// EO class Machine
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Inline methods implementation
//-----------------------------------------------------------------------------
void GateMachine::TestEndians () 
{
  int n = 1 ;
  char * p = (char *) & n ;
  m_Intel = (*p == (char) 1) ;
}

void GateMachine::TestBusSize () 
{
  if (sizeof(void *) == 4) 
    {
      m_32 = true ;
      m_64 = false ;
    } 
  else if (sizeof(void *) == 8) 
    {
      m_32 = false ;
      m_64 = true ;
    } 
  else 
    std::cout << "lgl::GateMachine::testBusSize : neither a 32 bits nor a 64 bits machine... Is it an extraterrestrial device ?" << std::endl ;
}


#define ENDIAN_INVERT_4BYTES(a) ((((unsigned int)a) >> 24) | ((a) << 24) | (((a) & 0x00FF0000) >> 8) | (((a) & 0x0000FF00) << 8))

void GateMachine::SwapEndians (unsigned short & d) { d = (d >> 8) | (d << 8);}
void GateMachine::SwapEndians (short & d) { d = (((unsigned short)d) >> 8) | (d << 8); }
void GateMachine::SwapEndians (unsigned int & d) {	d = ENDIAN_INVERT_4BYTES(d); }
void GateMachine::SwapEndians (int & d) { d = ENDIAN_INVERT_4BYTES(d); }
void GateMachine::SwapEndians (unsigned long & d) { SwapEndians(*((int*)&d)); }
void GateMachine::SwapEndians (long & d) {	SwapEndians(*((int*)&d)); }
void GateMachine::SwapEndians (float & d) { SwapEndians(*((int*)&d)); }
void GateMachine::SwapEndians (double & d) {
  int tmp = *((int*)&d);
  *((int*)&d) = ENDIAN_INVERT_4BYTES(*(((int*)&d)+1));
  *(((int*)&d)+1) = ENDIAN_INVERT_4BYTES(tmp);
}

void GateMachine::SwapEndians (unsigned short* d, long n) { for (long i=0;i<n;i++,++d) SwapEndians(*d); }
void GateMachine::SwapEndians (short* d, long n) { for (long i=0;i<n;i++,++d) SwapEndians(*d); }
void GateMachine::SwapEndians (unsigned int* d, long n) { for (long i=0;i<n;i++,++d) SwapEndians(*d); }
void GateMachine::SwapEndians (int* d, long n) { for (long i=0;i<n;i++,++d) SwapEndians(*d); }
void GateMachine::SwapEndians (unsigned long* d, long n) { for (long i=0;i<n;i++,++d) SwapEndians(*d); }
void GateMachine::SwapEndians (long* d, long n) { for (long i=0;i<n;i++,++d) SwapEndians(*d); }
void GateMachine::SwapEndians (float* d, long n) { for (long i=0;i<n;i++,++d) SwapEndians(*d); }
void GateMachine::SwapEndians (double* d, long n) { for (long i=0;i<n;i++,++d) SwapEndians(*d); }




//-----------------------------------------------------------------------------
// EOF
//-----------------------------------------------------------------------------
#endif
