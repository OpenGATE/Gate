/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/



#ifndef GateTokenizer_hh
#define GateTokenizer_hh

#include "globals.hh"
#include "G4Tokenizer.hh"
#include "GateConfiguration.h"

class GateTokenizer
: public G4Tokenizer
{
public:

  inline GateTokenizer(const G4String& itsString)
    : G4Tokenizer(itsString)
   {}

  virtual inline ~GateTokenizer() {}

public:
  struct BrokenString {
    G4String::size_type separatorPos;
    G4String first;
    G4String second;
    };


public:
  static void CleanUpString(G4String& stringToCleanup);
  static BrokenString BreakString(const G4String& stringToBreak,char separator);

public:
  G4SubString operator()(const char* str=" \t\n",size_t l=0)
    {
      return G4Tokenizer::operator()(str,l);
    }

};

#endif
