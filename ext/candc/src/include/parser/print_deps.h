/* -*- Mode: C++; -*- */
// C&C NLP tools
// Copyright (c) Universities of Edinburgh, Oxford and Sydney
// Copyright (c) James R. Curran
//
// This software is covered by a non-commercial use licence.
// See LICENCE.txt for the full text of the licence.
//
// If LICENCE.txt is not included in this distribution
// please email candc@it.usyd.edu.au to obtain a copy.

namespace NLP {
  namespace CCG {

    class DepsPrinter: public StreamPrinter {
    protected:
      virtual void unary(Sentence &sent){}
      virtual void derivation(const SuperCat *sc, Sentence &sent);
      virtual void lexical(Sentence &sent);
    public:
      DepsPrinter(Categories &cats, const Format FORMAT,
		  IO::Output &out, IO::Log &log)
	: StreamPrinter(cats, FORMAT, out, log){}

      virtual ~DepsPrinter(void){ /* do nothing */ }

      virtual void header(const std::string &preface);
      virtual void footer(void){}
    };
  }
}
