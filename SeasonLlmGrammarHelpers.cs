// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonLLM

namespace Season.LLM;

internal static class SeasonLlmGrammarHelpers
{
    private const string AnyJsonObjectGrammar = """
root ::= object
object ::= "{" space ( string ":" space value ( "," space string ":" space value )* )? "}" space
array ::= "[" space ( value ( "," space value )* )? "]" space
boolean ::= ( "true" | "false" ) space
char ::= [^"\\\x7F\x00-\x1F] | [\\] ( ["\\bfnrt] | "u" [0-9a-fA-F]{4} )
decimal-part ::= [0-9]{1,16}
integral-part ::= [0] | [1-9] [0-9]{0,15}
null ::= "null" space
number ::= ( "-"? integral-part ) ( "." decimal-part )? ( [eE] [-+]? integral-part )? space
string ::= "\"" char* "\"" space
value ::= object | array | string | number | boolean | null
space ::= "" | " " | "\n" [ \t]{0,20}
""";

    public static bool TryResolveGrammar(SeasonLlmGenerationOptions options, out string grammar, out string grammarRoot)
    {
        ArgumentNullException.ThrowIfNull(options);

        var hasGrammar = !string.IsNullOrWhiteSpace(options.Grammar);
        var hasJsonSchema = !string.IsNullOrWhiteSpace(options.JsonSchema);
        if ((hasGrammar ? 1 : 0) + (options.JsonOutput ? 1 : 0) + (hasJsonSchema ? 1 : 0) > 1)
        {
            throw new InvalidOperationException("Grammar, JsonOutput, and JsonSchema are mutually exclusive.");
        }

        if (hasGrammar)
        {
            grammar = options.Grammar!;
            grammarRoot = string.IsNullOrWhiteSpace(options.GrammarRoot) ? "root" : options.GrammarRoot;
            return true;
        }

        if (options.JsonOutput)
        {
            grammar = AnyJsonObjectGrammar;
            grammarRoot = "root";
            return true;
        }

        if (hasJsonSchema)
        {
            grammar = SeasonLlmJsonSchemaGrammarConverter.Convert(options.JsonSchema!);
            grammarRoot = "root";
            return true;
        }

        grammar = string.Empty;
        grammarRoot = string.Empty;
        return false;
    }
}

internal static class SeasonLlmJsonSchemaGrammarConverter
{
    public static string Convert(string jsonSchema)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(jsonSchema);

        using var document = JsonDocument.Parse(jsonSchema);
        if (document.RootElement.ValueKind != JsonValueKind.Object)
        {
            throw new NotSupportedException("JsonSchema must be a JSON object.");
        }

        return new Builder(document.RootElement).Build();
    }

    private sealed class Builder
    {
        private readonly JsonElement _rootSchema;
        private readonly List<KeyValuePair<string, string>> _rules = [];
        private readonly HashSet<string> _ruleNames = [];
        private readonly Dictionary<string, string> _schemaCache = [];
        private int _nextRuleId;
        private bool _needsSpaceRule;
        private bool _needsCharRule;
        private bool _needsIntegralPartRule;
        private bool _needsDecimalPartRule;
        private bool _needsNumberRule;
        private bool _needsStringRule;
        private bool _needsBooleanRule;
        private bool _needsNullRule;
        private bool _needsValueRule;
        private bool _needsObjectRule;
        private bool _needsArrayRule;

        public Builder(JsonElement rootSchema)
        {
            _rootSchema = rootSchema;
        }

        public string Build()
        {
            var rootRule = BuildSchemaRule(_rootSchema, "root");
            if (!string.Equals(rootRule, "root", StringComparison.Ordinal))
            {
                AddRule("root", rootRule);
            }

            AddHelperRules();

            var builder = new StringBuilder();
            foreach (var rule in _rules)
            {
                builder.Append(rule.Key);
                builder.Append(" ::= ");
                builder.Append(rule.Value);
                builder.AppendLine();
            }

            return builder.ToString();
        }

        private string BuildSchemaRule(JsonElement schema, string preferredName)
        {
            EnsureUnsupportedKeywordsAreAbsent(schema, ["$ref", "allOf", "anyOf", "oneOf", "not", "if", "then", "else", "dependentSchemas", "contains", "minContains", "maxContains", "patternProperties", "prefixItems"]);

            var cacheKey = schema.GetRawText();
            if (_schemaCache.TryGetValue(cacheKey, out var cachedRule))
            {
                return cachedRule;
            }

            string ruleName;
            if (TryGetProperty(schema, "const", out var constValue))
            {
                ruleName = CreateUniqueRuleName(preferredName);
                AddRule(ruleName, BuildLiteral(constValue));
            }
            else if (TryGetProperty(schema, "enum", out var enumValues))
            {
                if (enumValues.ValueKind != JsonValueKind.Array || enumValues.GetArrayLength() == 0)
                {
                    throw new NotSupportedException("JsonSchema enum must be a non-empty array.");
                }

                ruleName = CreateUniqueRuleName(preferredName);
                var alternatives = enumValues.EnumerateArray()
                    .Select(BuildLiteral)
                    .ToArray();
                AddRule(ruleName, string.Join(" | ", alternatives));
            }
            else
            {
                var type = ResolveSchemaType(schema);
                ruleName = type switch
                {
                    "object" => BuildObjectRule(schema, preferredName),
                    "array" => BuildArrayRule(schema, preferredName),
                    "string" => BuildStringRule(schema, preferredName),
                    "integer" => BuildIntegerRule(preferredName),
                    "number" => BuildNumberSchemaRule(preferredName),
                    "boolean" => BuildBooleanSchemaRule(preferredName),
                    "null" => BuildNullSchemaRule(preferredName),
                    _ => throw new NotSupportedException($"Unsupported JsonSchema type '{type}'.")
                };
            }

            _schemaCache[cacheKey] = ruleName;
            return ruleName;
        }

        private string BuildObjectRule(JsonElement schema, string preferredName)
        {
            var hasProperties = TryGetProperty(schema, "properties", out var propertiesElement);
            if (hasProperties && propertiesElement.ValueKind != JsonValueKind.Object)
            {
                throw new NotSupportedException("JsonSchema object properties must be an object.");
            }

            var allowsAdditionalProperties = false;
            if (TryGetProperty(schema, "additionalProperties", out var additionalProperties))
            {
                allowsAdditionalProperties = additionalProperties.ValueKind switch
                {
                    JsonValueKind.False => false,
                    JsonValueKind.True => throw new NotSupportedException("JsonSchema additionalProperties=true is not supported yet."),
                    JsonValueKind.Object => throw new NotSupportedException("JsonSchema additionalProperties schemas are not supported yet."),
                    _ => throw new NotSupportedException("JsonSchema additionalProperties must be a boolean or object.")
                };
            }

            if (!hasProperties || propertiesElement.GetRawText() == "{}")
            {
                if (!hasProperties && !TryGetProperty(schema, "additionalProperties", out _))
                {
                    _needsObjectRule = true;
                    return "object";
                }

                if (allowsAdditionalProperties)
                {
                    _needsObjectRule = true;
                    return "object";
                }

                var emptyRule = CreateUniqueRuleName(preferredName);
                AddRule(emptyRule, $"{Literal("{")} space {Literal("}")} space");
                _needsSpaceRule = true;
                return emptyRule;
            }

            var requiredNames = ReadRequiredProperties(schema);
            var properties = new List<PropertyRule>();
            foreach (var property in propertiesElement.EnumerateObject())
            {
                var valueRule = BuildSchemaRule(property.Value, $"{preferredName}-{property.Name}");
                var kvRule = CreateUniqueRuleName($"{preferredName}-{property.Name}-kv");
                AddRule(kvRule, $"{Literal(JsonSerializer.Serialize(property.Name))} space {Literal(":")} space {valueRule}");
                properties.Add(new PropertyRule(kvRule, requiredNames.Contains(property.Name)));
            }

            var bodyRule = BuildObjectBodyRule(properties, 0, hasWrittenProperty: false, $"{preferredName}-body");
            var objectRule = CreateUniqueRuleName(preferredName);
            AddRule(objectRule, $"{Literal("{")} space {bodyRule} {Literal("}")} space");
            _needsSpaceRule = true;
            return objectRule;
        }

        private string BuildObjectBodyRule(IReadOnlyList<PropertyRule> properties, int index, bool hasWrittenProperty, string preferredName)
        {
            var ruleName = CreateUniqueRuleName($"{preferredName}-{index}-{(hasWrittenProperty ? 1 : 0)}");
            if (index >= properties.Count)
            {
                AddRule(ruleName, "\"\"");
                return ruleName;
            }

            var property = properties[index];
            var includePrefix = hasWrittenProperty ? $"{Literal(",")} space " : string.Empty;
            var nextWhenIncluded = BuildObjectBodyRule(properties, index + 1, true, preferredName);
            if (property.Required)
            {
                AddRule(ruleName, $"{includePrefix}{property.RuleName} {nextWhenIncluded}");
                return ruleName;
            }

            var nextWhenSkipped = BuildObjectBodyRule(properties, index + 1, hasWrittenProperty, preferredName);
            AddRule(ruleName, $"{nextWhenSkipped} | {includePrefix}{property.RuleName} {nextWhenIncluded}");
            return ruleName;
        }

        private string BuildArrayRule(JsonElement schema, string preferredName)
        {
            EnsureUnsupportedKeywordsAreAbsent(schema, ["uniqueItems"]);

            var itemRule = TryGetProperty(schema, "items", out var itemsSchema)
                ? BuildSchemaRule(itemsSchema, $"{preferredName}-item")
                : BuildValueRule();

            var minItems = ReadNonNegativeInt(schema, "minItems");
            var maxItems = ReadOptionalNonNegativeInt(schema, "maxItems");
            if (maxItems.HasValue && maxItems.Value < minItems)
            {
                throw new NotSupportedException("JsonSchema maxItems cannot be smaller than minItems.");
            }

            var itemsExpression = BuildArrayItemsExpression(itemRule, minItems, maxItems);
            var ruleName = CreateUniqueRuleName(preferredName);
            AddRule(ruleName, $"{Literal("[")} space {itemsExpression} {Literal("]")} space");
            _needsSpaceRule = true;
            return ruleName;
        }

        private string BuildStringRule(JsonElement schema, string preferredName)
        {
            EnsureUnsupportedKeywordsAreAbsent(schema, ["pattern", "format"]);

            var minLength = ReadNonNegativeInt(schema, "minLength");
            var maxLength = ReadOptionalNonNegativeInt(schema, "maxLength");
            if (maxLength.HasValue && maxLength.Value < minLength)
            {
                throw new NotSupportedException("JsonSchema maxLength cannot be smaller than minLength.");
            }

            _needsCharRule = true;
            _needsSpaceRule = true;

            var repetition = maxLength.HasValue
                ? $"{{{minLength},{maxLength.Value}}}"
                : minLength == 0 ? "*" : $"{{{minLength},}}";

            var ruleName = CreateUniqueRuleName(preferredName);
            AddRule(ruleName, $"{Literal("\"")} char{repetition} {Literal("\"")} space");
            return ruleName;
        }

        private string BuildIntegerRule(string preferredName)
        {
            _needsIntegralPartRule = true;
            _needsSpaceRule = true;

            var ruleName = CreateUniqueRuleName(preferredName);
            AddRule(ruleName, $"( {Literal("-")}? integral-part ) space");
            return ruleName;
        }

        private string BuildNumberSchemaRule(string preferredName)
        {
            _needsNumberRule = true;
            return AddAliasRule(preferredName, "number");
        }

        private string BuildBooleanSchemaRule(string preferredName)
        {
            _needsBooleanRule = true;
            return AddAliasRule(preferredName, "boolean");
        }

        private string BuildNullSchemaRule(string preferredName)
        {
            _needsNullRule = true;
            return AddAliasRule(preferredName, "null");
        }

        private string BuildValueRule()
        {
            _needsValueRule = true;
            _needsObjectRule = true;
            _needsArrayRule = true;
            _needsStringRule = true;
            _needsNumberRule = true;
            _needsBooleanRule = true;
            _needsNullRule = true;
            _needsIntegralPartRule = true;
            _needsDecimalPartRule = true;
            _needsCharRule = true;
            _needsSpaceRule = true;
            return "value";
        }

        private static string ResolveSchemaType(JsonElement schema)
        {
            if (TryGetProperty(schema, "type", out var typeElement))
            {
                if (typeElement.ValueKind != JsonValueKind.String)
                {
                    throw new NotSupportedException("JsonSchema type must be a single string.");
                }

                return typeElement.GetString() ?? throw new NotSupportedException("JsonSchema type cannot be null.");
            }

            if (TryGetProperty(schema, "properties", out _))
            {
                return "object";
            }

            if (TryGetProperty(schema, "items", out _))
            {
                return "array";
            }

            return "object";
        }

        private static HashSet<string> ReadRequiredProperties(JsonElement schema)
        {
            var required = new HashSet<string>(StringComparer.Ordinal);
            if (!TryGetProperty(schema, "required", out var requiredElement))
            {
                return required;
            }

            if (requiredElement.ValueKind != JsonValueKind.Array)
            {
                throw new NotSupportedException("JsonSchema required must be an array of property names.");
            }

            foreach (var name in requiredElement.EnumerateArray())
            {
                if (name.ValueKind != JsonValueKind.String)
                {
                    throw new NotSupportedException("JsonSchema required entries must be strings.");
                }

                required.Add(name.GetString()!);
            }

            return required;
        }

        private static int ReadNonNegativeInt(JsonElement schema, string propertyName)
        {
            var value = ReadOptionalNonNegativeInt(schema, propertyName);
            return value ?? 0;
        }

        private static int? ReadOptionalNonNegativeInt(JsonElement schema, string propertyName)
        {
            if (!TryGetProperty(schema, propertyName, out var element))
            {
                return null;
            }

            if (!element.TryGetInt32(out var value) || value < 0)
            {
                throw new NotSupportedException($"JsonSchema {propertyName} must be a non-negative int32.");
            }

            return value;
        }

        private static void EnsureUnsupportedKeywordsAreAbsent(JsonElement schema, IReadOnlyCollection<string> names)
        {
            foreach (var name in names)
            {
                if (TryGetProperty(schema, name, out _))
                {
                    throw new NotSupportedException($"JsonSchema keyword '{name}' is not supported yet.");
                }
            }
        }

        private void AddHelperRules()
        {
            ResolveHelperDependencies();

            if (_needsCharRule)
            {
                AddRule("char", @"[^""\\\x7F\x00-\x1F] | [\\] ( [""\\bfnrt] | ""u"" [0-9a-fA-F]{4} )");
            }

            if (_needsDecimalPartRule)
            {
                AddRule("decimal-part", "[0-9]{1,16}");
            }

            if (_needsIntegralPartRule)
            {
                AddRule("integral-part", "[0] | [1-9] [0-9]{0,15}");
            }

            if (_needsNullRule)
            {
                AddRule("null", $"{Literal("null")} space");
            }

            if (_needsBooleanRule)
            {
                AddRule("boolean", $"( {Literal("true")} | {Literal("false")} ) space");
            }

            if (_needsNumberRule)
            {
                AddRule("number", $"( {Literal("-")}? integral-part ) ( {Literal(".")} decimal-part )? ( [eE] [-+]? integral-part )? space");
            }

            if (_needsStringRule)
            {
                AddRule("string", $"{Literal("\"")} char* {Literal("\"")} space");
            }

            if (_needsArrayRule)
            {
                AddRule("array", $"{Literal("[")} space ( value ( {Literal(",")} space value )* )? {Literal("]")} space");
            }

            if (_needsObjectRule)
            {
                AddRule("object", $"{Literal("{")} space ( string {Literal(":")} space value ( {Literal(",")} space string {Literal(":")} space value )* )? {Literal("}")} space");
            }

            if (_needsValueRule)
            {
                AddRule("value", "object | array | string | number | boolean | null");
            }

            if (_needsSpaceRule)
            {
                AddRule("space", "\"\" | \" \" | \"\\n\" [ \\t]{0,20}");
            }
        }

        private void ResolveHelperDependencies()
        {
            while (true)
            {
                var changed = false;
                changed |= SetHelperDependency(ref _needsObjectRule, _needsValueRule);
                changed |= SetHelperDependency(ref _needsArrayRule, _needsValueRule);
                changed |= SetHelperDependency(ref _needsStringRule, _needsValueRule);
                changed |= SetHelperDependency(ref _needsNumberRule, _needsValueRule);
                changed |= SetHelperDependency(ref _needsBooleanRule, _needsValueRule);
                changed |= SetHelperDependency(ref _needsNullRule, _needsValueRule);

                changed |= SetHelperDependency(ref _needsValueRule, _needsObjectRule);
                changed |= SetHelperDependency(ref _needsStringRule, _needsObjectRule);
                changed |= SetHelperDependency(ref _needsValueRule, _needsArrayRule);

                changed |= SetHelperDependency(ref _needsCharRule, _needsStringRule);
                changed |= SetHelperDependency(ref _needsIntegralPartRule, _needsNumberRule);
                changed |= SetHelperDependency(ref _needsDecimalPartRule, _needsNumberRule);

                changed |= SetHelperDependency(ref _needsSpaceRule, _needsNullRule);
                changed |= SetHelperDependency(ref _needsSpaceRule, _needsBooleanRule);
                changed |= SetHelperDependency(ref _needsSpaceRule, _needsNumberRule);
                changed |= SetHelperDependency(ref _needsSpaceRule, _needsStringRule);
                changed |= SetHelperDependency(ref _needsSpaceRule, _needsArrayRule);
                changed |= SetHelperDependency(ref _needsSpaceRule, _needsObjectRule);

                if (!changed)
                {
                    return;
                }
            }
        }

        private static bool SetHelperDependency(ref bool target, bool shouldEnable)
        {
            if (!shouldEnable || target)
            {
                return false;
            }

            target = true;
            return true;
        }

        private string AddAliasRule(string preferredName, string targetRule)
        {
            var ruleName = CreateUniqueRuleName(preferredName);
            AddRule(ruleName, targetRule);
            return ruleName;
        }

        private string BuildArrayItemsExpression(string itemRule, int minItems, int? maxItems)
        {
            if (maxItems == 0)
            {
                return "\"\"";
            }

            var tailRule = $"{Literal(",")} space {itemRule}";
            if (!maxItems.HasValue)
            {
                return minItems == 0
                    ? $"\"\" | {itemRule} ( {tailRule} )*"
                    : $"{itemRule} ( {tailRule} ){{{minItems - 1},}}";
            }

            if (minItems == 0)
            {
                return $"\"\" | {itemRule} ( {tailRule} ){{0,{maxItems.Value - 1}}}";
            }

            return $"{itemRule} ( {tailRule} ){{{minItems - 1},{maxItems.Value - 1}}}";
        }

        private static string BuildLiteral(JsonElement value)
        {
            return $"{Literal(JsonSerializer.Serialize(value))} space";
        }

        private static bool TryGetProperty(JsonElement element, string propertyName, out JsonElement value)
        {
            if (element.ValueKind == JsonValueKind.Object && element.TryGetProperty(propertyName, out value))
            {
                return true;
            }

            value = default;
            return false;
        }

        private static string Literal(string text)
        {
            var builder = new StringBuilder(text.Length + 8);
            builder.Append('"');
            foreach (var ch in text)
            {
                switch (ch)
                {
                    case '\\':
                        builder.Append(@"\\");
                        break;
                    case '"':
                        builder.Append("\\\"");
                        break;
                    case '\n':
                        builder.Append(@"\n");
                        break;
                    case '\r':
                        builder.Append(@"\r");
                        break;
                    case '\t':
                        builder.Append(@"\t");
                        break;
                    default:
                        builder.Append(ch);
                        break;
                }
            }

            builder.Append('"');
            return builder.ToString();
        }

        private string CreateUniqueRuleName(string preferredName)
        {
            var sanitized = SanitizeRuleName(preferredName);
            if (_ruleNames.Add(sanitized))
            {
                return sanitized;
            }

            while (true)
            {
                var candidate = $"{sanitized}-{_nextRuleId++}";
                if (_ruleNames.Add(candidate))
                {
                    return candidate;
                }
            }
        }

        private void AddRule(string name, string expression)
        {
            if (_rules.Any(pair => pair.Key == name))
            {
                return;
            }

            _rules.Add(new KeyValuePair<string, string>(name, expression));
        }

        private static string SanitizeRuleName(string preferredName)
        {
            if (string.IsNullOrWhiteSpace(preferredName))
            {
                return "rule";
            }

            var builder = new StringBuilder(preferredName.Length);
            var previousDash = false;
            foreach (var ch in preferredName)
            {
                var normalized = char.ToLowerInvariant(ch);
                if ((normalized >= 'a' && normalized <= 'z') || (normalized >= '0' && normalized <= '9'))
                {
                    builder.Append(normalized);
                    previousDash = false;
                }
                else if (!previousDash)
                {
                    builder.Append('-');
                    previousDash = true;
                }
            }

            var value = builder.ToString().Trim('-');
            if (value.Length == 0)
            {
                return "rule";
            }

            if (value[0] >= '0' && value[0] <= '9')
            {
                value = $"rule-{value}";
            }

            return value;
        }

        private readonly record struct PropertyRule(string RuleName, bool Required);
    }
}
