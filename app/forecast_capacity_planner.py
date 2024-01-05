import numpy as np
import plotly.graph_objects as go

class ForecastCapPlan:
    
    def __init__(self, city, week, log_data, billable_data, nonbillable_data, config):
        
        # Configuration File
        self.config = config
        
        # Specify City
        self.City = city
        
        # Specify Business Week
        self.BusinessWeek = week
        
        # Load in Log Data
        self.log_data = log_data.loc[week]
        self.ActiveCount = float(self.log_data['Active Unique Agents per Week'])
        self.TargetCount = float(self.log_data['Required FTE'])
        
        
        # Load in Billable Data
        self.billable_data = billable_data.loc[week]
        self.billable_activity_breakdown()
        
        # Load in NonBillable Data
        self.nonbillable_data = nonbillable_data.loc[week]
        self.LeaveAllocation = float(self.nonbillable_data['LeaveCount'] + self.nonbillable_data['Absenteeism Count']) 
        self.nonbillable_activity_breakdown()    
        
        # Run Aggergation
        self.aggregated_activity_breakdown()
        self.KPI_calculation()
    
    def billable_activity_breakdown(self):
        self.available = self.billable_data.get("available Hours", 0)
        self.onboarding = self.billable_data.get("onboarding Hours", 0)
        self.coaching = self.billable_data.get("coaching Hours", 0)
        self.meeting = self.billable_data.get("team_meeting Hours", 0)
        self.wellness = self.billable_data.get("wellness_support Hours", 0)
        self.fbtraining = self.billable_data.get("fb_training Hours", 0)
        
    def nonbillable_activity_breakdown(self):
        self.meal = self.nonbillable_data.get("Meal Hours",0)
        self.breaks = self.nonbillable_data.get("Break Hours", 0)
        self.leaves = self.nonbillable_data.get("Leave Hours",0) + self.nonbillable_data.get("Planned Absenteeism Hours",0)
        self.nonfbtraining = self.nonbillable_data.get("Non-Meta Training", 0)
        
    def aggregated_activity_breakdown(self):
        self.Productive = sum([getattr(self, x, 0) for x in self.config['simulator']['activityTypes']['productive']])
        self.Billable = sum([getattr(self, x, 0) for x in self.config['simulator']['activityTypes']['billable']])
        self.Nonbillable = sum([getattr(self, x, 0) for x in self.config['simulator']['activityTypes']['nonbillable']])
    
    def KPI_calculation(self):
        self.TotalHours = self.Billable + self.Nonbillable
        self.RequiredBillableHours = (self.TargetCount * (1-self.config['simulator']['OOO_Target'])) * (self.config['CityWorkHours'][self.City]['client'] * 5) * (.851)
        
        self.WIO = (self.Billable - self.Productive)/(self.Billable)
        self.OOO = self.Nonbillable/self.TotalHours
        self.UtilizationInternal = self.Productive/self.Billable # Based on Internal FTE Count
        self.UtilizationClient = self.Productive/self.RequiredBillableHours# Based on Client FTE Target 
        self.BillableTarget = self.Billable/self.RequiredBillableHours # How many percent did we exceed/lack
        
    def generate_waterfall(self):
        import plotly.graph_objects as go
        fig = go.Figure(
            go.Waterfall(
                name = "20",
                orientation = "v",
                measure = [
                    "relative",
                    "relative","relative","relative","relative","relative",
                    "relative",
                    "relative", "relative",
                    "total",
                    "relative",
                    "total",
                    "relative",
                    "total"
                ],
                x = [
                    "Productive",
                    "Onboarding", "Coaching", "Meeting", "Meta Training", "Wellness Support",
                    "Non-Meta Training",
                    "Leave Hours",
                    "Break Hours",
                    "Logged Hours",
                    "OOO Shrinkage",
                    "Billable Hours",
                    "WIO Shrinkage",
                    "Utilization Hours"
                ],
                y = [
                    self.Productive,
                    self.onboarding, self.coaching, self.meeting, self.fbtraining, self.wellness,
                    self.nonfbtraining,
                    self.leaves,
                    self.meal + self.breaks, 
                    0,
                    -1*self.Nonbillable,
                    0,
                    -1*(self.Billable-self.Productive),
                    0
                ],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
                textposition = "outside",
                showlegend = False,
                text = [
                    f"+{100*self.Productive/self.TotalHours:.2f}%",
                    
                    f"+{100*self.onboarding/self.TotalHours:.2f}%",
                    f"+{100*self.coaching/self.TotalHours:.2f}%",
                    f"+{100*self.meeting/self.TotalHours:.2f}%",
                    f"+{100*self.fbtraining/self.TotalHours:.2f}%",
                    f"+{100*self.wellness/self.TotalHours:.2f}%",
                    f"+{100*self.nonfbtraining/self.TotalHours:.2f}%",
                    
                    f"+{100*self.leaves/self.TotalHours:.2f}%", 
                    f"+{100*(self.meal + self.breaks)/self.TotalHours:.2f}%",  
                    "",
                    #f"{100*(self.meal + self.breaks + self.leaves)/self.TotalHours:.2f}%",
                    f"{100*self.OOO:.2f}%",
                    "",
                    #f"{100*(self.Billable-self.Productive)/self.TotalHours:.2f}%",
                    f"{100*self.WIO:.2f}%",
                    ""
                ],
                customdata = np.stack(
                    (
                        [self.ActiveCount] * 14, # Current Headcount
                        [self.TargetCount] * 14, # Target FTE
                        [self.LeaveAllocation] * 14, # Leave Allocation
                        [
                            100 * self.Productive/self.TotalHours,
                            
                            100 * self.onboarding/self.TotalHours, 
                            100 * self.coaching/self.TotalHours, 
                            100 * self.meeting/self.TotalHours, 
                            100 * self.fbtraining/self.TotalHours, 
                            100 * self.wellness/self.TotalHours,
                            100 * self.nonfbtraining/self.TotalHours,
                            
                            100 * self.leaves/self.TotalHours,
                            100 * (self.meal + self.breaks)/self.TotalHours,
                            
                            100 * (self.TotalHours)/self.TotalHours,
                            100 * self.OOO,
                            100 * (1-self.OOO),
                            100 * self.WIO,
                            100 * self.Productive/self.TotalHours
                        ],
                        [
                            self.Productive,
                            
                            self.onboarding, 
                            self.coaching, 
                            self.meeting, 
                            self.fbtraining,
                            self.wellness,
                            self.nonfbtraining,
                            
                            self.leaves,
                            (self.meal + self.breaks),
                            
                            (self.TotalHours),
                            self.Nonbillable,
                            (self.TotalHours - self.Nonbillable),
                            (self.Billable - self.Productive),
                            self.Productive
                        ],
                        [
                            100 * self.Productive/self.Billable,
                            
                            100 * self.onboarding/self.Billable, 
                            100 * self.coaching/self.Billable, 
                            100 * self.meeting/self.Billable, 
                            100 * self.fbtraining/self.Billable, 
                            100 * self.wellness/self.Billable,
                            100 * self.nonfbtraining/self.Billable,
                            
                            0,
                            0,
                            
                            0,
                            0,
                            0,
                            100 * -1 * (self.Productive - self.Billable)/self.Billable,
                            100 * self.Productive/self.Billable
                        ],
                        [
                            "",
                            
                            "", 
                            "", 
                            "", 
                            "",
                            "",
                            "",
                            
                            f"(Annual Leaves + TOI Leaves + Absenteeism Allocation) x {self.config['CityWorkHours'][self.City]['internal']}<br>",
                            "",

                            "Total Logged Hours<br>",
                            "Leaves + Breaks + Meals<br>",
                            "Total Logged Hours - (Leaves + Breaks + Meals)<br>",
                            "Total Billable Hours - Productive Hours<br>",
                            ""
                        ],
                        
                    ),
                    axis = 1
                ),
                hovertemplate=
                "<b>%{x}</b><br>" +
                "%{customdata[6]}"+
                
                "<br><b>Activity Info</b><br>" +
                "Activity Hours: %{customdata[4]:.2f}<br>" +
                "% of Total Hours: %{customdata[3]:.2f}%<br>" +
                
                "<br><b>HeadCount Info</b><br>" +
                "Actual FTE: %{customdata[0]:.0f}<br>" +
                "Target FTE: %{customdata[1]:.1f}<br>"+
                "Leaves Allocation: %{customdata[2]:.2f}<br>"+
                
                "<br><b>% Allocation wrt Logged Billable Hours:</b><br>" +
                "%{x} : %{customdata[5]:.2f}%"+
                "<extra></extra>",
            )
        )
        
        fig.add_shape(type='line',
                        x0=8.5,
                        y0=self.TotalHours,
                        x1=11.5,
                        y1=self.TotalHours,
                        line=dict(color='Red',),
                        xref='x',
                        yref='y'
        )
        
        fig.add_shape(type='line',
                        x0=5.5,
                        y0=self.TotalHours,
                        x1=13.5,
                        y1=self.TotalHours,
                        line=dict(color='Red',),
                        xref='x',
                        yref='y'
        )
        
        fig.add_shape(type='line',
                        x0=0.5,
                        y0=self.Billable,
                        x1=15.5,
                        y1=self.Billable,
                        line=dict(color='Red',),
                        xref='x',
                        yref='y'
        )
        
        fig.update_layout(
            autosize=True,
            width=1600,
            height=800,)
        
        return fig
        