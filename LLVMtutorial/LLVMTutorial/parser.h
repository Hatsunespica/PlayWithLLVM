#ifndef PARSER_H_INCLUDED
#define PARSER_H_INCLUDED

#include "AST.h"
#include <string>
#include <map>
#include <memory>

using namespace tutorial;

enum Token{
           tok_eof=-1,
           tok_def=-2,
           tok_extern=-3,
           tok_identifier=-4,
           tok_number=-5,
};

static std::string identifierStr;
static double numVal;

int gettok();


extern int curTok;
int getNextToken();

extern std::map<char,int> binopPrecedence;

int getTokPrecedence();
std::unique_ptr<ExprAST> parseExpression();
std::unique_ptr<ExprAST> parseNumberExpr();
std::unique_ptr<ExprAST> parseParenExpr();
std::unique_ptr<ExprAST> parseIdentifierExpr();
std::unique_ptr<ExprAST> parsePrimary();
std::unique_ptr<ExprAST> parseBinopRHS();
std::unique_ptr<ExprAST> parseExpression();
std::unique_ptr<PrototypeAST> parsePrototype();
std::unique_ptr<FunctionAST> parseDefinition();
std::unique_ptr<FunctionAST> parseTopLevelExpr();
std::unique_ptr<PrototypeAST> parseExtern();

#endif
