/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateTokenizer.hh"


//---------------------------------------------------------------------------
// Go through the whole DB file from the start
// until we find a specific text (at beginning of line)
// Cleans-up a string by removing both leading and trailing
// spaces and tabulations
void GateTokenizer::CleanUpString(G4String& stringToCleanup)
{
  while (stringToCleanup.length()) {
    if ( (stringToCleanup.at(0)==' ') || (stringToCleanup.at(0)=='\t') || (stringToCleanup.at(0)=='\n') )
      stringToCleanup=stringToCleanup.substr(1);
    else
      break;
  }
  while (stringToCleanup.length()) {
    G4String::size_type pos = stringToCleanup.length()-1;
    if ( (stringToCleanup.at(pos)==' ') || (stringToCleanup.at(pos)=='\t') || (stringToCleanup.at(0)=='\n') )
      stringToCleanup=stringToCleanup.substr(0,pos);
    else
      break;
  }
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
// Break a string into 2 components separated by a predefined separator
// Returns a structure BrokenString with:
//  separatorPos: position of the separator in the original string
//  first: everything before the separator (separator itself excluded)
//  second: everything after the separator (separator itself excluded)
GateTokenizer::BrokenString GateTokenizer::BreakString(const G4String& stringToBreak,char separator)
{
  BrokenString stringPair;

  // Find separator position("=")
  G4String::size_type separatorPos=stringToBreak.find_first_of(separator);
  if (separatorPos==G4String::npos) {
    stringPair.separatorPos = G4String::npos;
    stringPair.first =  stringToBreak;
    stringPair.second = "";
  } else {
    stringPair.separatorPos = separatorPos;
    stringPair.first =  stringToBreak.substr(0,separatorPos);
    stringPair.second = stringToBreak.substr(separatorPos+1);
  }
  return stringPair;
}
//---------------------------------------------------------------------------
