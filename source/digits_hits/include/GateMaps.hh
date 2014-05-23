/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateMaps_hh
#define GateMaps_hh

#include "globals.hh"
#include <map>

/*! \class  GateMap
    \brief  A general templates for enhanced maps (i.e. maps with a few added functions)

    - GateMap - by Daniel.Strul@iphe.unil.ch

    - GateMaps are standard STL maps with some added functionalities for easier construction and printout
*/
template <typename Tkey,typename Tvalue>
class GateMap : public std::map<Tkey,Tvalue>
{
public:
  /*! \class MapPair
      \brief A GateMap::MapPair is an element of a GateMap
  */
  typedef     	   std::pair<Tkey,Tvalue>   	      MapPair ;
  typedef typename std::map<Tkey,Tvalue>::iterator  iterator ;

  //! Defaut constructor -> empty GateMap
  GateMap();

  //! Constructs a GateMap with one item
  GateMap(MapPair* first);

  //! Constructs a GateMap from an array of MapPairs
  //! In agremment with the STL map template, the arguments are the start and the end of the array
  //! Remember that the end is one step AFTER the last element of the array
  //! For instance, for an array 'array' with 3 elements, call : GateMap(array,array+3)
  GateMap(MapPair* first,MapPair* last);

  //! Constructs a GateMap from an array of MapPairs
  //! Pass the size of the array a pointer to and its first element
  GateMap(size_t n, MapPair* first);

  //! Constructs a new GateMap by combining several GateMaps
  //! Pass the number of GateMaps and an array of GateMap pointers
  GateMap(size_t n, GateMap<Tkey,Tvalue> *mapArray[]);

  // Destructor
  virtual inline ~GateMap() {}


public:
  //! Dump a map into a printable string
  //! This function concatenates all the map keys into one printable string
  //! Each key can be preceded by a prefix and followed by a suffix. It can be printed within quotes.
  //! Also, each key can be separated from the next one by a separator-string.
  virtual G4String DumpMap(G4bool flagWithQuotes=false,G4String suffix="",G4String separator=" ",G4String prefix="" );

};


// Defaut constructor -> empty GateMap
template <typename Tkey,typename Tvalue>
inline GateMap<Tkey,Tvalue>::GateMap()
 : std::map<Tkey,Tvalue>()
{}




// Constructs a GateMap with one item
template <typename Tkey,typename Tvalue>
inline GateMap<Tkey,Tvalue>::GateMap(MapPair* first)
 : std::map<Tkey,Tvalue>(first,first+1)
{}




// Constructs a GateMap from an array of MapPairs
// In agremment with the STL map template, the arguments are the start and the end of the array
// Remember that the end is one step AFTER the last element of the array
// For instance, for an array 'array' with 3 elements, call : GateMap(array,array+3)
template <typename Tkey,typename Tvalue>
inline GateMap<Tkey,Tvalue>::GateMap(MapPair* first,MapPair* last)
 : std::map<Tkey,Tvalue>(first,last)
{}





// Constructs a GateMap from an array of MapPairs
// Pass the size of the array a pointer to and its first element
template <typename Tkey,typename Tvalue>
inline GateMap<Tkey,Tvalue>::GateMap(size_t n, MapPair* first)
 : std::map<Tkey,Tvalue>(first,first+n)
{}





// Constructs a new GateMap by combining several GateMaps
// Pass the number of GateMaps and an array of GateMap pointers
template <typename Tkey,typename Tvalue>
inline GateMap<Tkey,Tvalue>::GateMap(size_t n, GateMap<Tkey,Tvalue> *mapArray[])
 : std::map<Tkey,Tvalue>()
{
  for (size_t i=0; i<n ; i++){
    GateMap<Tkey,Tvalue>* mapElement = mapArray[i];
    for (iterator iter = mapElement->begin(); iter != mapElement->end(); iter++)
      this->insert(*iter);
  }
}




// Dump a map into a printable string
// This function concatenates all the map keys into one printable string
// Each key can be preceded by a prefix and followed by a suffix. It can be printed within quotes.
// Also, each key can be separated from the next one by a separator-string.
template <typename Tkey,typename Tvalue>
inline G4String GateMap<Tkey,Tvalue>::DumpMap(G4bool flagWithQuotes, G4String suffix, G4String separator, G4String prefix )
{
    G4String dump;

    iterator iter = this->begin();
    while (iter != this->end()) {
      dump += prefix;
      if (flagWithQuotes)
      	dump += "'";
      dump += G4String(iter->first) + suffix;
      if (flagWithQuotes)
      	dump += "'";
       if ( (++iter) != this->end() )
      	dump += separator;
    }
    return dump;
}




/*! \class GateUnitMap
    \brief A GateUnitMap is a map of unit-name/unit-value pairs
*/
typedef GateMap<G4String,G4double> GateUnitMap;




/*! \class GateUnitPair
    \brief A GateUnitPair combines a unit name and a unit value
*/
typedef GateUnitMap::MapPair GateUnitPair;




/*! \class GateCodeMap
    \brief A GateCodeMap is a map of codename/codevalue pairs
*/
typedef GateMap<G4String,G4int> GateCodeMap;




/*! \class GateCodePair
    \brief A GateCodePair combines a string (code-name) and an integer (code-value)
*/
typedef GateCodeMap::MapPair GateCodePair;



#endif
