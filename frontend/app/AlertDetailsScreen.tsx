import React from "react";
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from "react-native";
import { colors } from "../theme/colors";

// Helper function to dynamically determine risk color and icon
const getRiskDetails = (risk: string) => {
  switch (risk.toLowerCase()) {
    case 'high':
      return {
        color: '#DC2626',
        icon: '🚨',
        title: 'CRITICAL ALERT'
      };
    case 'medium':
      return {
        color: '#D97706',
        icon: '⚠️',
        title: 'WARNING ALERT'
      };
    case 'low':
      return {
        color: '#2563EB',
        icon: 'ℹ️',
        title: 'LOW RISK INFO'
      };
    default:
      return {
        color: '#6B7280',
        icon: '❓',
        title: 'UNKNOWN RISK'
      };
  }
};

export default function AlertDetailsScreen({ route }: any) {
  const { alert } = route.params;
  const riskDetails = getRiskDetails(alert.risk);

  const suggestedActions = [
    "Verify the resident did NOT share personal or financial info.",
    "Contact the resident's family or guardian immediately.",
    "Report suspicious phone numbers or messages to FTC.",
    "Reset any potentially compromised passwords.",
  ];

  return (
    <ScrollView style={styles.container}>
      {/* 1. COLOR-CODED HEADER BAR */}
      <View style={[styles.riskHeader, { backgroundColor: riskDetails.color }]}>
        <Text style={styles.riskIcon}>{riskDetails.icon}</Text>
        <Text style={styles.riskTitle}>{riskDetails.title}</Text>
        <Text style={styles.riskLevel}>{alert.risk.toUpperCase()}</Text>
      </View>

      <View style={styles.content}>
        {/* 2. ALERT DETAILS CARD */}
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>Case Information</Text>
          
          <View style={styles.detailRow}>
            <Text style={styles.label}>Resident:</Text>
            <Text style={styles.value}>{alert.resident}</Text>
          </View>
          
          <View style={styles.detailRow}>
            <Text style={styles.label}>Alert Type:</Text>
            <Text style={styles.value}>{alert.type}</Text>
          </View>
          
          <View style={styles.detailRow}>
            <Text style={styles.label}>Time Detected:</Text>
            <Text style={styles.value}>10:30 AM, Nov 18</Text> 
          </View>

        </View>

        {/* 3. SUGGESTED ACTIONS LIST */}
        <Text style={styles.sectionTitle}>Next Steps & Actions</Text>
        <View style={styles.actionsCard}>
            {suggestedActions.map((action, index) => (
                <View key={index} style={styles.actionItem}>
                    <Text style={styles.bullet}>•</Text>
                    <Text style={styles.actionText}>{action}</Text>
                </View>
            ))}
        </View>
        
        {/* 4. ACTION BUTTONS */}
        <TouchableOpacity style={[styles.primaryButton, { backgroundColor: riskDetails.color }]}>
          <Text style={styles.primaryButtonText}>
            RESOLVE & MARK SAFE
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.secondaryButton}>
          <Text style={styles.secondaryButtonText}>
            Contact Resident Guardian
          </Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#FDF2F8",
  },
  content: {
    padding: 20,
  },
  
  // --- Risk Header Bar ---
  riskHeader: {
    paddingVertical: 30,
    paddingHorizontal: 20,
    alignItems: 'center',
    marginBottom: 20,
  },
  riskIcon: {
      fontSize: 40,
      marginBottom: 5,
  },
  riskTitle: {
    fontSize: 22,
    fontWeight: '900',
    color: '#FFFFFF',
    letterSpacing: 1,
  },
  riskLevel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
    opacity: 0.8,
  },

  // --- Cards & Details ---
  sectionTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: "#1F2937",
    marginBottom: 10,
    marginTop: 10,
  },
  card: {
    backgroundColor: "#FFFFFF",
    padding: 20,
    borderRadius: 20,
    marginBottom: 20,
    elevation: 4,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  detailRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      paddingVertical: 8,
      borderBottomWidth: 1,
      borderBottomColor: "#F3F4F6",
  },
  label: {
    fontSize: 14,
    color: "#6B7280",
    fontWeight: '600',
  },
  value: {
    fontSize: 16,
    fontWeight: "800",
    color: "#1F2937",
  },
  
  // --- Actions List ---
  actionsCard: {
    backgroundColor: "#FFFFFF",
    padding: 20,
    borderRadius: 20,
    marginBottom: 30,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
  },
  actionItem: {
      flexDirection: 'row',
      marginBottom: 10,
  },
  bullet: {
      fontSize: 16,
      color: "#DC2626",
      marginRight: 10,
  },
  actionText: {
    flex: 1,
    fontSize: 15,
    color: "#1F2937",
    lineHeight: 22,
  },
  
  // --- Buttons ---
  primaryButton: {
    paddingVertical: 18,
    borderRadius: 30,
    marginBottom: 15,
    elevation: 5,
  },
  primaryButtonText: {
    fontSize: 16,
    fontWeight: "900",
    color: "#FFFFFF",
    textAlign: 'center',
    textTransform: 'uppercase',
  },
  secondaryButton: {
    backgroundColor: 'transparent',
    paddingVertical: 18,
    marginBottom: 20,
  },
  secondaryButtonText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#6B7280",
    textAlign: 'center',
    textDecorationLine: 'underline',
  },
});