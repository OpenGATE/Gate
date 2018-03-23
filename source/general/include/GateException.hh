/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/



/**
 *  \file 
 *  \brief  class Exception:generic class for throwing any exception (header) 
 *
 *    Long description:
 */

/**
 *  \class Gate::Exception 
 *  \brief  class Exception : generic class for throwing any exception 
 *
 *    Long description:
 */
 
#ifndef __GateException_h__
#define __GateException_h__

//#include "GateSystem.h"
#include <exception>

namespace Gate
{

  inline std::string bbGetObjectDescription() { return(""); }

  class Exception : public std::exception
  {
  public:
    Exception(const std::string& object,
	      const std::string& file,
	      const std::string& message) throw()
      : mObject(object),
	mFile(file),
	mMessage(message)
    {}
    ~Exception() throw() {}
    void Print() throw()
    {
      std::cout << "* ERROR  : " << mMessage << Gateendl; 
      //       printf("ERROR : [%s]\n",mLabel.c_str());
      int lev = GateMessageManager::GetMessageLevel("Error");
      if (lev > 0) {
	std::cout << "* OBJECT : " <<mObject<< Gateendl;
	std::cout << "* FILE   : " <<mFile<< Gateendl;
	//More info displayed
      }
    }
    const std::string& GetObject() const { return mObject; }
    const std::string& GetFile() const { return mFile; }
    const std::string& GetMessage() const { return mMessage; }
  private:
    std::string mObject;
    std::string mFile;
    std::string mMessage;
  };

}//namespace

#endif
