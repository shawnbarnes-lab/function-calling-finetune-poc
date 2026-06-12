# Inference Comparison: Base vs Fine-tuned (Qwen3-32B)

- **base**: `qwen3-32b-base`
- **finetuned**: `qwen3-32b-xlam-lora`

## Test set: 20 prompts

## Aggregate results

| Endpoint | Valid function calls | Avg latency (ms) | Success |
|---|---|---|---|
| base | 3/20 | 483 | 20/20 |
| finetuned | 16/20 | 503 | 20/20 |

---

## Prompt 1 - single-call

**User:** What is the weather like in Detroit right now?

### base (764ms) [invalid]

```
```json
{
  "function": "get_weather",
  "parameters": {
    "location": "Detroit, MI"
  }
}
```
```

### finetuned (1130ms) [valid function call]

```
[{"name":"get_weather","arguments":{"location":"Detroit, MI","unit":"fahrenheit"}}]
```

---

## Prompt 2 - single-call

**User:** Find me flights from Detroit to New York on December 15th

### base (625ms) [valid function call]

```
{
  "name": "search_flights",
  "parameters": {
    "origin": "Detroit",
    "destination": "New York",
    "date": "2021-12-15"
  }
}
```

### finetuned (569ms) [valid function call]

```
[{"name":"search_flights","arguments":{"origin":"Detroit","destination":"New York","date":"2022-12-15"}}]
```

---

## Prompt 3 - single-call

**User:** What is 1247 multiplied by 33?

### base (441ms) [invalid]

```
```json
{
  "function": "calculate",
  "parameters": {
    "expression": "1247 * 33"
  }
}
```
```

### finetuned (360ms) [valid function call]

```
[{"name":"calculate","arguments":{"expression":"1247 * 33"}}]
```

---

## Prompt 4 - single-call

**User:** Send an email to alexei@nebius.com with subject Demo and body Looking forward to it

### base (654ms) [invalid]

```
```json
{
  "function": "send_email",
  "parameters": {
    "to": "alexei@nebius.com",
    "subject": "Demo",
    "body": "Looking forward to it"
  }
}
```
```

### finetuned (547ms) [valid function call]

```
[{"name":"send_email","arguments":{"to":"alexei@nebius.com","subject":"Demo","body":"Looking forward to it"}}]
```

---

## Prompt 5 - multi-call

**User:** Check the weather in Paris and Tokyo, then tell me which is warmer

### base (836ms) [invalid]

```
{
  "function_calls": [
    {
      "name": "get_weather",
      "parameters": {
        "location": "Paris, France"
      }
    },
    {
      "name": "get_weather",
      "parameters": {
        "location": "Tokyo, Japan"
      }
    }
  ]
}
```

### finetuned (630ms) [valid function call]

```
[{"name":"get_weather","arguments":{"location":"Paris","unit":"celsius"}},{"name":"get_weather","arguments":{"location":"Tokyo","unit":"celsius"}}]
```

---

## Prompt 6 - ambiguous

**User:** What's the temperature?

### base (229ms) [invalid]

```
Could you please specify the location for which you'd like to know the temperature?
```

### finetuned (49ms) [invalid]

```
[]
```

---

## Prompt 7 - no-call-needed

**User:** What is the capital of France?

### base (114ms) [invalid]

```
The capital of France is Paris.
```

### finetuned (347ms) [invalid]

```
I cannot provide the capital of France as it is not within the scope of the functions provided.
```

---

## Prompt 8 - single-call

**User:** Find flights from SFO to JFK tomorrow

### base (684ms) [valid function call]

```
```json
{
  "name": "search_flights",
  "parameters": {
    "origin": "SFO",
    "destination": "JFK",
    "date": "2023-09-26"
  }
}
```
```

### finetuned (587ms) [valid function call]

```
[{"name":"search_flights","arguments":{"origin":"SFO","destination":"JFK","date":"2022-01-01"}}]
```

---

## Prompt 9 - single-call

**User:** Whats 250 plus 175 minus 30

### base (299ms) [invalid]

```
The result of 250 plus 175 minus 30 is 395.
```

### finetuned (421ms) [valid function call]

```
[{"name":"calculate","arguments":{"expression":"250 + 175 - 30"}}]
```

---

## Prompt 10 - single-call

**User:** Weather in Seattle in fahrenheit please

### base (491ms) [invalid]

```
```json
{
  "function": "get_weather",
  "parameters": {
    "location": "Seattle, WA",
    "unit": "fahrenheit"
  }
}
```
```

### finetuned (379ms) [valid function call]

```
[{"name":"get_weather","arguments":{"location":"Seattle, WA","unit":"fahrenheit"}}]
```

---

## Prompt 11 - multi-call

**User:** Email mom@example.com with subject Hi and tell her I'll be there at 6pm. Also check the weather in Chicago.

### base (1112ms) [invalid]

```
```json
{
  "function_calls": [
    {
      "name": "send_email",
      "parameters": {
        "to": "mom@example.com",
        "subject": "Hi",
        "body": "I'll be there at 6pm."
      }
    },
    {
      "name": "get_weather",
      "parameters": {
        "location": "Chicago, IL"
      }
    }
  ]
}
```
```

### finetuned (829ms) [valid function call]

```
[{"name":"send_email","arguments":{"to":"mom@example.com","subject":"Hi","body":"I'll be there at 6pm"}},{"name":"get_weather","arguments":{"location":"Chicago, IL","unit":"fahrenheit"}}]
```

---

## Prompt 12 - ambiguous

**User:** Book me a flight

### base (253ms) [invalid]

```
Could you please provide the origin city, destination city, and the date for your flight?
```

### finetuned (653ms) [valid function call]

```
[{"name":"search_flights","arguments":{"origin":"New York, NY","destination":"Los Angeles, CA","date":"2022-01-01"}}]
```

---

## Prompt 13 - no-call-needed

**User:** Who wrote Hamlet?

### base (121ms) [invalid]

```
William Shakespeare wrote *Hamlet*.
```

### finetuned (380ms) [invalid]

```
I cannot provide the answer to your question as it does not fall under the functions I have access to.
```

---

## Prompt 14 - single-call

**User:** Compute the square root of 144

### base (189ms) [invalid]

```
The square root of 144 is 12.
```

### finetuned (330ms) [valid function call]

```
[{"name":"calculate","arguments":{"expression":"sqrt(144)"}}]
```

---

## Prompt 15 - single-call

**User:** Whats the forecast for Austin TX

### base (383ms) [invalid]

```
```json
{
  "function": "get_weather",
  "parameters": {
    "location": "Austin TX"
  }
}
```
```

### finetuned (381ms) [valid function call]

```
[{"name":"get_weather","arguments":{"location":"Austin, TX","unit":"fahrenheit"}}]
```

---

## Prompt 16 - multi-call

**User:** Find flights from BOS to MIA on Dec 20 and email me the results at me@example.com

### base (625ms) [valid function call]

```
{
  "name": "search_flights",
  "parameters": {
    "origin": "BOS",
    "destination": "MIA",
    "date": "2021-12-20"
  }
}
```

### finetuned (1101ms) [valid function call]

```
[{"name":"search_flights","arguments":{"origin":"BOS","destination":"MIA","date":"2022-12-20"}},{"name":"send_email","arguments":{"to":"me@example.com","subject":"Flight Search Results","body":"Here are the flight search results for your query."}}]
```

---

## Prompt 17 - edge-case

**User:** Calculate 0/0

### base (354ms) [invalid]

```
The expression $ 0/0 $ is undefined in mathematics, as division by zero is not a valid operation.
```

### finetuned (283ms) [valid function call]

```
[{"name":"calculate","arguments":{"expression":"0/0"}}]
```

---

## Prompt 18 - single-call

**User:** Send a quick note to team@example.com about the meeting being moved to 3pm

### base (748ms) [invalid]

```
```json
{
  "function": "send_email",
  "parameters": {
    "to": "team@example.com",
    "subject": "Meeting Time Update",
    "body": "The meeting has been moved to 3pm. Please adjust your schedules accordingly."
  }
}
```
```

### finetuned (595ms) [valid function call]

```
[{"name":"send_email","arguments":{"to":"team@example.com","subject":"Meeting Time Change","body":"The meeting has been moved to 3pm."}}]
```

---

## Prompt 19 - single-call

**User:** What is 2^32

### base (357ms) [invalid]

```
The result of $ 2^{32} $ is 4,294,967,296.
```

### finetuned (298ms) [valid function call]

```
[{"name":"calculate","arguments":{"expression":"2^32"}}]
```

---

## Prompt 20 - no-call-needed

**User:** Tell me a joke

### base (383ms) [invalid]

```
Sure! Here's a light-hearted one for you:

Why don't skeletons fight each other?

Because they don't have the guts! 😄
```

### finetuned (199ms) [invalid]

```
I'm sorry, but I cannot tell jokes.
```

---

