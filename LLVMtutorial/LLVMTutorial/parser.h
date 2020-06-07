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
           tok_if=-6,
           tok_then=-7,
           tok_else=-8,
           tok_for=-9,
           tok_in=-10,
           tok_binary=-11,
           tok_unary=-12,
};

static std::string identifierStr;
static double numVal;

int gettok();


extern int curTok;
int getNextToken();

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
std::unique_ptr<ExprAST> parseIfExpr();
std::unique_ptr<ExprAST> parseForExpr();
std::unique_ptr<ExprAST> parseUnary();

#endif
